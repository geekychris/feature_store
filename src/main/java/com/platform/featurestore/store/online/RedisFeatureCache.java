package com.platform.featurestore.store.online;

import com.platform.featurestore.proto.FeatureVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * Redis hot cache layer implementing cache-aside pattern.
 * <p>
 * On read: check Redis → fallback to RocksDB → populate Redis
 * On write: write to RocksDB → update Redis
 * <p>
 * Values stored as Protobuf byte arrays for compact serialization.
 */
@Component
@ConditionalOnProperty(name = "feature-store.redis.enabled", havingValue = "true", matchIfMissing = true)
public class RedisFeatureCache {

    private static final Logger log = LoggerFactory.getLogger(RedisFeatureCache.class);

    private final RedisTemplate<String, byte[]> redisTemplate;
    private final Duration cacheTtl;
    private final String keyPrefix;

    @Autowired
    public RedisFeatureCache(RedisTemplate<String, byte[]> featureStoreRedisTemplate,
                              Duration redisCacheTtl,
                              String redisKeyPrefix) {
        this.redisTemplate = featureStoreRedisTemplate;
        this.cacheTtl = redisCacheTtl;
        this.keyPrefix = redisKeyPrefix;
    }

    /**
     * Get a cached feature vector.
     */
    public Optional<FeatureVector> getVector(String viewName, int viewVersion, String entityId) {
        String key = vectorCacheKey(viewName, viewVersion, entityId);
        try {
            byte[] raw = redisTemplate.opsForValue().get(key);
            if (raw == null) {
                return Optional.empty();
            }
            return Optional.of(FeatureVector.parseFrom(raw));
        } catch (Exception e) {
            log.warn("Redis cache read failed for {}: {}", key, e.getMessage());
            return Optional.empty();
        }
    }

    /**
     * Cache a feature vector with TTL.
     */
    public void putVector(String viewName, int viewVersion, String entityId,
                           FeatureVector vector) {
        String key = vectorCacheKey(viewName, viewVersion, entityId);
        try {
            redisTemplate.opsForValue().set(key, vector.toByteArray(), cacheTtl);
        } catch (Exception e) {
            log.warn("Redis cache write failed for {}: {}", key, e.getMessage());
        }
    }

    /**
     * Batch get vectors from cache.
     * Returns a map of entityId → FeatureVector for cache hits only.
     */
    public Map<String, FeatureVector> getVectorsBatch(String viewName, int viewVersion,
                                                       List<String> entityIds) {
        List<String> keys = entityIds.stream()
                .map(eid -> vectorCacheKey(viewName, viewVersion, eid))
                .toList();

        Map<String, FeatureVector> result = new LinkedHashMap<>();
        try {
            List<byte[]> values = redisTemplate.opsForValue().multiGet(keys);
            if (values == null) return result;

            for (int i = 0; i < entityIds.size(); i++) {
                byte[] raw = values.get(i);
                if (raw != null) {
                    try {
                        result.put(entityIds.get(i), FeatureVector.parseFrom(raw));
                    } catch (com.google.protobuf.InvalidProtocolBufferException e) {
                        log.warn("Failed to parse cached vector for {}", entityIds.get(i));
                    }
                }
            }
        } catch (Exception e) {
            log.warn("Redis batch read failed: {}", e.getMessage());
        }
        return result;
    }

    /**
     * Invalidate a cached vector.
     */
    public void invalidateVector(String viewName, int viewVersion, String entityId) {
        String key = vectorCacheKey(viewName, viewVersion, entityId);
        try {
            redisTemplate.delete(key);
        } catch (Exception e) {
            log.warn("Redis cache invalidation failed for {}: {}", key, e.getMessage());
        }
    }

    /**
     * Cache key format: fs:v:{viewName}:{version}:{entityId}
     */
    private String vectorCacheKey(String viewName, int viewVersion, String entityId) {
        return keyPrefix + "v:" + viewName + ":" + viewVersion + ":" + entityId;
    }
}
