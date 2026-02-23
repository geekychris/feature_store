package com.platform.featurestore.store.online;

import com.platform.featurestore.config.RocksDBConfig;
import com.platform.featurestore.proto.Feature;
import com.platform.featurestore.proto.FeatureValue;
import com.platform.featurestore.proto.FeatureVector;
import com.platform.featurestore.proto.ViewSchema;
import org.rocksdb.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;

/**
 * Online serving layer backed by RocksDB.
 * <p>
 * Column family strategy:
 *   feature_vectors  — pre-materialized flattened vectors (primary fast path)
 *   feature_scalars  — individual scalar values (fallback + partial updates)
 *   embeddings       — raw float arrays
 *   schemas          — view schema registry cache
 */
@Component
public class RocksDBFeatureStore {

    private static final Logger log = LoggerFactory.getLogger(RocksDBFeatureStore.class);

    private final RocksDB db;
    private final Map<String, ColumnFamilyHandle> cfHandles;

    public RocksDBFeatureStore(RocksDB db, Map<String, ColumnFamilyHandle> columnFamilyHandles) {
        this.db = db;
        this.cfHandles = columnFamilyHandles;
    }

    // ------------------------------------------------------------------
    // Key construction (matches Python design)
    // ------------------------------------------------------------------

    /**
     * Vector key: [view_name_hash:4B][view_version:2B][entity_id:varB]
     */
    public byte[] vectorKey(String viewName, int viewVersion, String entityId) {
        byte[] viewHash = hashPrefix(viewName);
        byte[] entityBytes = entityId.getBytes(StandardCharsets.UTF_8);

        ByteBuffer buf = ByteBuffer.allocate(4 + 2 + entityBytes.length);
        buf.order(ByteOrder.BIG_ENDIAN);
        buf.put(viewHash);
        buf.putShort((short) viewVersion);
        buf.put(entityBytes);
        return buf.array();
    }

    /**
     * Scalar key: [entity_type_hash:4B][entity_id:varB][NUL:1B][feature_name:varB]
     */
    public byte[] scalarKey(String entityType, String entityId, String featureName) {
        byte[] typeHash = hashPrefix(entityType);
        byte[] idBytes = entityId.getBytes(StandardCharsets.UTF_8);
        byte[] nameBytes = featureName.getBytes(StandardCharsets.UTF_8);

        ByteBuffer buf = ByteBuffer.allocate(4 + idBytes.length + 1 + nameBytes.length);
        buf.order(ByteOrder.BIG_ENDIAN);
        buf.put(typeHash);
        buf.put(idBytes);
        buf.put((byte) 0);  // NUL separator
        buf.put(nameBytes);
        return buf.array();
    }

    /**
     * Schema key: [view_name:varB][NUL:1B][version_str:varB]
     */
    public byte[] schemaKey(String viewName, int version) {
        String key = viewName + "\0" + version;
        return key.getBytes(StandardCharsets.UTF_8);
    }

    /**
     * First 4 bytes of MD5 hash — used as key prefix for locality.
     */
    private byte[] hashPrefix(String input) {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] digest = md.digest(input.getBytes(StandardCharsets.UTF_8));
            return Arrays.copyOf(digest, 4);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("MD5 not available", e);
        }
    }

    // ------------------------------------------------------------------
    // Write path
    // ------------------------------------------------------------------

    public void putFeatureVector(String viewName, int viewVersion, String entityId,
                                  FeatureVector vector) throws RocksDBException {
        byte[] key = vectorKey(viewName, viewVersion, entityId);
        byte[] value = vector.toByteArray();
        db.put(cfHandles.get(RocksDBConfig.CF_FEATURE_VECTORS), key, value);
    }

    public void putScalarFeature(String entityType, String entityId,
                                  Feature feature) throws RocksDBException {
        byte[] key = scalarKey(entityType, entityId, feature.getName());
        byte[] value = feature.toByteArray();
        db.put(cfHandles.get(RocksDBConfig.CF_FEATURE_SCALARS), key, value);
    }

    public void putEmbedding(String entityType, String entityId, String featureName,
                              float[] embedding, int version) throws RocksDBException {
        byte[] key = scalarKey(entityType, entityId, featureName);

        ByteBuffer buf = ByteBuffer.allocate(4 + embedding.length * 4);
        buf.order(ByteOrder.BIG_ENDIAN);
        buf.putShort((short) version);
        buf.putShort((short) embedding.length);
        for (float f : embedding) {
            buf.putFloat(f);
        }

        db.put(cfHandles.get(RocksDBConfig.CF_EMBEDDINGS), key, buf.array());
    }

    public void putSchema(ViewSchema schema) throws RocksDBException {
        byte[] key = schemaKey(schema.getViewName(), schema.getVersion());
        byte[] value = schema.toByteArray();
        db.put(cfHandles.get(RocksDBConfig.CF_SCHEMAS), key, value);
    }

    /**
     * Batch write scalars using WriteBatch for atomicity and throughput.
     */
    public void putScalarBatch(String entityType,
                                List<ScalarRecord> records) throws RocksDBException {
        ColumnFamilyHandle cf = cfHandles.get(RocksDBConfig.CF_FEATURE_SCALARS);
        try (WriteBatch batch = new WriteBatch();
             WriteOptions writeOpts = new WriteOptions()) {
            for (ScalarRecord record : records) {
                byte[] key = scalarKey(entityType, record.entityId(), record.featureName());
                Feature feature = Feature.newBuilder()
                        .setName(record.featureName())
                        .setValue(FeatureValue.newBuilder()
                                .setFloat64Val(record.value())
                                .build())
                        .setEventTimeMs(System.currentTimeMillis())
                        .build();
                batch.put(cf, key, feature.toByteArray());
            }
            db.write(writeOpts, batch);
        }
    }

    /**
     * Batch write feature vectors using WriteBatch.
     */
    public void putFeatureVectorBatch(List<FeatureVector> vectors) throws RocksDBException {
        ColumnFamilyHandle cf = cfHandles.get(RocksDBConfig.CF_FEATURE_VECTORS);
        try (WriteBatch batch = new WriteBatch();
             WriteOptions writeOpts = new WriteOptions()) {
            for (FeatureVector vector : vectors) {
                byte[] key = vectorKey(vector.getViewName(), vector.getViewVersion(),
                        vector.getEntityId());
                batch.put(cf, key, vector.toByteArray());
            }
            db.write(writeOpts, batch);
        }
    }

    // ------------------------------------------------------------------
    // Read path
    // ------------------------------------------------------------------

    /**
     * Primary fast path — single key lookup returns ready-to-use vector.
     */
    public Optional<FeatureVector> getFeatureVector(String viewName, int viewVersion,
                                                     String entityId) throws RocksDBException {
        byte[] key = vectorKey(viewName, viewVersion, entityId);
        byte[] raw = db.get(cfHandles.get(RocksDBConfig.CF_FEATURE_VECTORS), key);
        if (raw == null) {
            return Optional.empty();
        }
        try {
            return Optional.of(FeatureVector.parseFrom(raw));
        } catch (com.google.protobuf.InvalidProtocolBufferException e) {
            log.error("Failed to deserialize FeatureVector for {}/{}/{}", viewName, viewVersion, entityId, e);
            return Optional.empty();
        }
    }

    /**
     * MultiGet — fetches N vectors in a single optimized call.
     */
    public Map<String, FeatureVector> getFeatureVectorsBatch(String viewName, int viewVersion,
                                                              List<String> entityIds) throws RocksDBException {
        ColumnFamilyHandle cf = cfHandles.get(RocksDBConfig.CF_FEATURE_VECTORS);
        List<ColumnFamilyHandle> cfs = new ArrayList<>();
        List<byte[]> keys = new ArrayList<>();

        for (String eid : entityIds) {
            cfs.add(cf);
            keys.add(vectorKey(viewName, viewVersion, eid));
        }

        List<byte[]> results = db.multiGetAsList(cfs, keys);
        Map<String, FeatureVector> map = new LinkedHashMap<>();

        for (int i = 0; i < entityIds.size(); i++) {
            byte[] raw = results.get(i);
            if (raw != null) {
                try {
                    map.put(entityIds.get(i), FeatureVector.parseFrom(raw));
                } catch (com.google.protobuf.InvalidProtocolBufferException e) {
                    log.error("Failed to deserialize vector for {}", entityIds.get(i), e);
                }
            }
        }
        return map;
    }

    public Optional<Feature> getScalarFeature(String entityType, String entityId,
                                               String featureName) throws RocksDBException {
        byte[] key = scalarKey(entityType, entityId, featureName);
        byte[] raw = db.get(cfHandles.get(RocksDBConfig.CF_FEATURE_SCALARS), key);
        if (raw == null) {
            return Optional.empty();
        }
        try {
            return Optional.of(Feature.parseFrom(raw));
        } catch (com.google.protobuf.InvalidProtocolBufferException e) {
            log.error("Failed to deserialize Feature for {}/{}/{}", entityType, entityId, featureName, e);
            return Optional.empty();
        }
    }

    public Optional<float[]> getEmbedding(String entityType, String entityId,
                                           String featureName) throws RocksDBException {
        byte[] key = scalarKey(entityType, entityId, featureName);
        byte[] raw = db.get(cfHandles.get(RocksDBConfig.CF_EMBEDDINGS), key);
        if (raw == null) {
            return Optional.empty();
        }

        ByteBuffer buf = ByteBuffer.wrap(raw).order(ByteOrder.BIG_ENDIAN);
        int version = buf.getShort() & 0xFFFF;
        int dim = buf.getShort() & 0xFFFF;
        float[] embedding = new float[dim];
        for (int i = 0; i < dim; i++) {
            embedding[i] = buf.getFloat();
        }
        return Optional.of(embedding);
    }

    public Optional<ViewSchema> getSchema(String viewName, int version) throws RocksDBException {
        byte[] key = schemaKey(viewName, version);
        byte[] raw = db.get(cfHandles.get(RocksDBConfig.CF_SCHEMAS), key);
        if (raw == null) {
            return Optional.empty();
        }
        try {
            return Optional.of(ViewSchema.parseFrom(raw));
        } catch (com.google.protobuf.InvalidProtocolBufferException e) {
            log.error("Failed to deserialize ViewSchema for {}/{}", viewName, version, e);
            return Optional.empty();
        }
    }

    /**
     * Fallback path: assemble vector from individual scalars via MultiGet.
     * Used when pre-materialized vector is missing or stale.
     */
    public FeatureVector assembleVectorFromScalars(String entityType, String entityId,
                                                    ViewSchema schema,
                                                    Map<String, Double> defaults) throws RocksDBException {
        ColumnFamilyHandle cf = cfHandles.get(RocksDBConfig.CF_FEATURE_SCALARS);
        List<ColumnFamilyHandle> cfs = new ArrayList<>();
        List<byte[]> keys = new ArrayList<>();

        for (String fname : schema.getFeatureNamesList()) {
            cfs.add(cf);
            keys.add(scalarKey(entityType, entityId, fname));
        }

        List<byte[]> results = db.multiGetAsList(cfs, keys);
        long nowMs = System.currentTimeMillis();

        FeatureVector.Builder builder = FeatureVector.newBuilder()
                .setViewName(schema.getViewName())
                .setViewVersion(schema.getVersion())
                .setEntityType(entityType)
                .setEntityId(entityId)
                .setServedAtMs(nowMs)
                .setSchemaHash(schema.getSchemaHash());

        for (int i = 0; i < schema.getFeatureNamesList().size(); i++) {
            String fname = schema.getFeatureNames(i);
            byte[] raw = results.get(i);

            if (raw == null) {
                builder.addValues(defaults.getOrDefault(fname, 0.0));
                builder.addIsDefaultMask(true);
                builder.addValueAgesMs(-1L);
            } else {
                try {
                    Feature feature = Feature.parseFrom(raw);
                    builder.addValues(extractFloat(feature.getValue()));
                    builder.addIsDefaultMask(false);
                    long age = feature.getEventTimeMs() > 0 ? nowMs - feature.getEventTimeMs() : -1;
                    builder.addValueAgesMs(age);
                } catch (com.google.protobuf.InvalidProtocolBufferException e) {
                    builder.addValues(defaults.getOrDefault(fname, 0.0));
                    builder.addIsDefaultMask(true);
                    builder.addValueAgesMs(-1L);
                }
            }
        }

        return builder.build();
    }

    /**
     * Extract a double from any numeric FeatureValue variant.
     */
    public static double extractFloat(FeatureValue fv) {
        return switch (fv.getValueCase()) {
            case FLOAT64_VAL -> fv.getFloat64Val();
            case FLOAT32_VAL -> fv.getFloat32Val();
            case INT64_VAL -> (double) fv.getInt64Val();
            case INT32_VAL -> (double) fv.getInt32Val();
            case BOOL_VAL -> fv.getBoolVal() ? 1.0 : 0.0;
            default -> 0.0;
        };
    }

    // ------------------------------------------------------------------
    // Record types
    // ------------------------------------------------------------------

    public record ScalarRecord(String entityId, String featureName, double value) {}
}
