package com.platform.featurestore.service;

import com.platform.featurestore.proto.Feature;
import com.platform.featurestore.proto.FeatureVector;
import com.platform.featurestore.store.offline.IcebergOfflineStore;
import com.platform.featurestore.store.online.RocksDBFeatureStore;
import com.platform.featurestore.store.online.RedisFeatureCache;
import org.rocksdb.RocksDBException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * Materializes features from source data into all storage layers.
 */
@Service
public class MaterializationService {

    private static final Logger log = LoggerFactory.getLogger(MaterializationService.class);

    private final RocksDBFeatureStore rocksStore;
    private final RedisFeatureCache redisCache;
    private final IcebergOfflineStore offlineStore;

    @Autowired
    public MaterializationService(RocksDBFeatureStore rocksStore,
                                   @Autowired(required = false) RedisFeatureCache redisCache,
                                   IcebergOfflineStore offlineStore) {
        this.rocksStore = rocksStore;
        this.redisCache = redisCache;
        this.offlineStore = offlineStore;
    }

    /**
     * Materialize a single feature vector to online stores.
     */
    public void materializeVector(FeatureVector vector) throws RocksDBException {
        // Write to RocksDB
        rocksStore.putFeatureVector(
                vector.getViewName(), vector.getViewVersion(),
                vector.getEntityId(), vector);

        // Update Redis cache
        if (redisCache != null) {
            redisCache.putVector(
                    vector.getViewName(), vector.getViewVersion(),
                    vector.getEntityId(), vector);
        }
    }

    /**
     * Materialize a batch of feature vectors.
     */
    public int materializeVectorBatch(List<FeatureVector> vectors) throws RocksDBException {
        // Batch write to RocksDB
        rocksStore.putFeatureVectorBatch(vectors);

        // Update Redis cache
        if (redisCache != null) {
            for (FeatureVector v : vectors) {
                redisCache.putVector(v.getViewName(), v.getViewVersion(),
                        v.getEntityId(), v);
            }
        }

        log.info("Materialized {} vectors to online stores", vectors.size());
        return vectors.size();
    }

    /**
     * Materialize scalar features to online store.
     */
    public void materializeScalars(String entityType,
                                    List<RocksDBFeatureStore.ScalarRecord> records) throws RocksDBException {
        rocksStore.putScalarBatch(entityType, records);
        log.info("Materialized {} scalar features for entity type {}", records.size(), entityType);
    }

    /**
     * Write feature records to the offline store for training data.
     */
    public void materializeToOffline(List<IcebergOfflineStore.FeatureRecordData> records) throws IOException {
        offlineStore.writeFeatureRecords(records);
        log.info("Materialized {} records to offline store", records.size());
    }

    /**
     * Full materialization: write to both online and offline stores.
     */
    public MaterializationResult materializeFull(String viewName, int viewVersion,
                                                   String entityType,
                                                   List<FeatureVector> vectors) throws RocksDBException, IOException {
        int onlineCount = materializeVectorBatch(vectors);

        // Also write to offline store
        List<IcebergOfflineStore.FeatureRecordData> offlineRecords = new ArrayList<>();
        OffsetDateTime now = OffsetDateTime.now();

        for (FeatureVector vector : vectors) {
            for (int i = 0; i < vector.getValuesCount(); i++) {
                offlineRecords.add(new IcebergOfflineStore.FeatureRecordData(
                        entityType,
                        vector.getEntityId(),
                        "feature_" + i,  // Will be resolved by schema
                        vector.getValues(i),
                        null,
                        now,
                        now,
                        "materialization_service",
                        viewVersion
                ));
            }
        }

        offlineStore.writeFeatureRecords(offlineRecords);

        return new MaterializationResult(onlineCount, offlineRecords.size());
    }

    public record MaterializationResult(int onlineVectors, int offlineRecords) {}
}
