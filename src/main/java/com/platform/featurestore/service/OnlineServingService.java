package com.platform.featurestore.service;

import com.platform.featurestore.proto.FeatureVector;
import com.platform.featurestore.proto.ViewSchema;
import com.platform.featurestore.store.online.RocksDBFeatureStore;
import com.platform.featurestore.store.online.RedisFeatureCache;
import org.rocksdb.RocksDBException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;

/**
 * Online serving service: orchestrates the read path.
 * <p>
 * Read path priority:
 * 1. Redis cache (hot path, sub-ms)
 * 2. RocksDB pre-materialized vector (fast path, ~1ms)
 * 3. RocksDB scalar assembly via MultiGet (fallback, ~5ms)
 */
@Service
public class OnlineServingService {

    private static final Logger log = LoggerFactory.getLogger(OnlineServingService.class);

    private final RocksDBFeatureStore rocksStore;
    private final RedisFeatureCache redisCache;

    @Autowired
    public OnlineServingService(RocksDBFeatureStore rocksStore,
                                 @Autowired(required = false) RedisFeatureCache redisCache) {
        this.rocksStore = rocksStore;
        this.redisCache = redisCache;
    }

    /**
     * Get a single feature vector with the full read path.
     */
    public ServingResult getFeatureVector(String viewName, int viewVersion, String entityId) {
        long start = System.nanoTime();
        List<String> warnings = new ArrayList<>();

        // 1. Try Redis cache
        if (redisCache != null) {
            Optional<FeatureVector> cached = redisCache.getVector(viewName, viewVersion, entityId);
            if (cached.isPresent()) {
                long latencyUs = (System.nanoTime() - start) / 1000;
                return new ServingResult(cached.get(), latencyUs, warnings, "REDIS_CACHE");
            }
        }

        // 2. Try RocksDB pre-materialized vector
        try {
            Optional<FeatureVector> vector = rocksStore.getFeatureVector(viewName, viewVersion, entityId);
            if (vector.isPresent()) {
                FeatureVector fv = vector.get();

                // Check staleness
                checkStaleness(fv, warnings);

                // Populate Redis cache for next time
                if (redisCache != null) {
                    redisCache.putVector(viewName, viewVersion, entityId, fv);
                }

                long latencyUs = (System.nanoTime() - start) / 1000;
                return new ServingResult(fv, latencyUs, warnings, "ROCKSDB_VECTOR");
            }
        } catch (RocksDBException e) {
            log.error("RocksDB read failed for vector {}/{}/{}", viewName, viewVersion, entityId, e);
            warnings.add("RocksDB vector read failed: " + e.getMessage());
        }

        // 3. Fallback: assemble from scalars
        try {
            Optional<ViewSchema> schema = rocksStore.getSchema(viewName, viewVersion);
            if (schema.isPresent()) {
                FeatureVector assembled = rocksStore.assembleVectorFromScalars(
                        "merchant", entityId, schema.get(), Map.of());

                // Cache the assembled vector
                if (redisCache != null) {
                    redisCache.putVector(viewName, viewVersion, entityId, assembled);
                }

                warnings.add("Vector assembled from scalars (not pre-materialized)");
                long latencyUs = (System.nanoTime() - start) / 1000;
                return new ServingResult(assembled, latencyUs, warnings, "ROCKSDB_SCALAR_ASSEMBLY");
            }
        } catch (RocksDBException e) {
            log.error("RocksDB scalar assembly failed for {}/{}/{}", viewName, viewVersion, entityId, e);
            warnings.add("Scalar assembly failed: " + e.getMessage());
        }

        long latencyUs = (System.nanoTime() - start) / 1000;
        return new ServingResult(null, latencyUs, warnings, "MISS");
    }

    /**
     * Batch get feature vectors.
     */
    public BatchServingResult getFeatureVectorsBatch(String viewName, int viewVersion,
                                                      List<String> entityIds) {
        long start = System.nanoTime();
        List<String> warnings = new ArrayList<>();
        Map<String, FeatureVector> results = new LinkedHashMap<>();

        // Separate into cache hits and misses
        Set<String> remaining = new LinkedHashSet<>(entityIds);

        // 1. Try Redis batch
        if (redisCache != null) {
            Map<String, FeatureVector> cached = redisCache.getVectorsBatch(
                    viewName, viewVersion, entityIds);
            results.putAll(cached);
            remaining.removeAll(cached.keySet());
        }

        // 2. RocksDB batch for remaining
        if (!remaining.isEmpty()) {
            try {
                Map<String, FeatureVector> rocksResults = rocksStore.getFeatureVectorsBatch(
                        viewName, viewVersion, new ArrayList<>(remaining));
                results.putAll(rocksResults);

                // Populate cache for fetched vectors
                if (redisCache != null) {
                    rocksResults.forEach((eid, fv) ->
                            redisCache.putVector(viewName, viewVersion, eid, fv));
                }

                remaining.removeAll(rocksResults.keySet());
            } catch (RocksDBException e) {
                log.error("RocksDB batch read failed", e);
                warnings.add("RocksDB batch read failed: " + e.getMessage());
            }
        }

        if (!remaining.isEmpty()) {
            warnings.add("Missing vectors for " + remaining.size() + " entities");
        }

        long latencyUs = (System.nanoTime() - start) / 1000;
        List<FeatureVector> vectors = entityIds.stream()
                .map(results::get)
                .filter(Objects::nonNull)
                .toList();

        return new BatchServingResult(vectors, latencyUs, warnings);
    }

    private void checkStaleness(FeatureVector vector, List<String> warnings) {
        long nowMs = System.currentTimeMillis();
        long ageMs = nowMs - vector.getServedAtMs();
        if (ageMs > 86400_000) { // 24 hours
            warnings.add("Vector is " + (ageMs / 3600_000) + " hours old");
        }
    }

    // Result records

    public record ServingResult(
            FeatureVector vector,
            long latencyUs,
            List<String> warnings,
            String source
    ) {}

    public record BatchServingResult(
            List<FeatureVector> vectors,
            long latencyUs,
            List<String> warnings
    ) {}
}
