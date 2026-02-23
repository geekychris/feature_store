package com.platform.featurestore.store.online;

import com.platform.featurestore.config.RocksDBConfig;
import com.platform.featurestore.proto.*;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;
import org.rocksdb.*;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class RocksDBFeatureStoreTest {

    @TempDir
    Path tempDir;

    private RocksDB db;
    private Map<String, ColumnFamilyHandle> cfHandles;
    private RocksDBFeatureStore store;
    private List<ColumnFamilyHandle> handles;

    @BeforeEach
    void setUp() throws RocksDBException {
        RocksDB.loadLibrary();

        List<ColumnFamilyDescriptor> cfDescriptors = new ArrayList<>();
        cfDescriptors.add(new ColumnFamilyDescriptor(RocksDB.DEFAULT_COLUMN_FAMILY));
        cfDescriptors.add(new ColumnFamilyDescriptor("feature_vectors".getBytes()));
        cfDescriptors.add(new ColumnFamilyDescriptor("feature_scalars".getBytes()));
        cfDescriptors.add(new ColumnFamilyDescriptor("embeddings".getBytes()));
        cfDescriptors.add(new ColumnFamilyDescriptor("schemas".getBytes()));

        DBOptions opts = new DBOptions()
                .setCreateIfMissing(true)
                .setCreateMissingColumnFamilies(true);

        handles = new ArrayList<>();
        db = RocksDB.open(opts, tempDir.toString(), cfDescriptors, handles);

        cfHandles = Map.of(
                "default", handles.get(0),
                RocksDBConfig.CF_FEATURE_VECTORS, handles.get(1),
                RocksDBConfig.CF_FEATURE_SCALARS, handles.get(2),
                RocksDBConfig.CF_EMBEDDINGS, handles.get(3),
                RocksDBConfig.CF_SCHEMAS, handles.get(4)
        );

        store = new RocksDBFeatureStore(db, cfHandles);
    }

    @AfterEach
    void tearDown() {
        handles.forEach(ColumnFamilyHandle::close);
        db.close();
    }

    @Test
    void testPutAndGetFeatureVector() throws RocksDBException {
        FeatureVector vector = FeatureVector.newBuilder()
                .setViewName("test_view")
                .setViewVersion(1)
                .setEntityType("merchant")
                .setEntityId("m_001")
                .addValues(0.008)
                .addValues(45230.0)
                .addValues(423.0)
                .setSchemaHash(12345)
                .setServedAtMs(System.currentTimeMillis())
                .build();

        store.putFeatureVector("test_view", 1, "m_001", vector);

        Optional<FeatureVector> result = store.getFeatureVector("test_view", 1, "m_001");
        assertTrue(result.isPresent());
        assertEquals("m_001", result.get().getEntityId());
        assertEquals(3, result.get().getValuesCount());
        assertEquals(0.008, result.get().getValues(0), 1e-6);
        assertEquals(45230.0, result.get().getValues(1), 1e-6);
        assertEquals(12345, result.get().getSchemaHash());
    }

    @Test
    void testGetMissingVector() throws RocksDBException {
        Optional<FeatureVector> result = store.getFeatureVector("nonexistent", 1, "m_999");
        assertTrue(result.isEmpty());
    }

    @Test
    void testBatchPutAndGet() throws RocksDBException {
        List<FeatureVector> vectors = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            vectors.add(FeatureVector.newBuilder()
                    .setViewName("batch_view")
                    .setViewVersion(1)
                    .setEntityType("merchant")
                    .setEntityId("m_" + String.format("%04d", i))
                    .addValues(i * 1.0)
                    .addValues(i * 2.0)
                    .setSchemaHash(99)
                    .setServedAtMs(System.currentTimeMillis())
                    .build());
        }

        store.putFeatureVectorBatch(vectors);

        // Batch read
        List<String> ids = List.of("m_0000", "m_0050", "m_0099", "m_9999");
        Map<String, FeatureVector> results = store.getFeatureVectorsBatch("batch_view", 1, ids);

        assertEquals(3, results.size()); // m_9999 should be missing
        assertTrue(results.containsKey("m_0000"));
        assertTrue(results.containsKey("m_0050"));
        assertTrue(results.containsKey("m_0099"));
        assertFalse(results.containsKey("m_9999"));

        assertEquals(0.0, results.get("m_0000").getValues(0), 1e-6);
        assertEquals(50.0, results.get("m_0050").getValues(0), 1e-6);
    }

    @Test
    void testScalarFeatures() throws RocksDBException {
        Feature feature = Feature.newBuilder()
                .setName("chargeback_rate")
                .setValue(FeatureValue.newBuilder().setFloat64Val(0.008).build())
                .setEventTimeMs(System.currentTimeMillis())
                .build();

        store.putScalarFeature("merchant", "m_001", feature);

        Optional<Feature> result = store.getScalarFeature("merchant", "m_001", "chargeback_rate");
        assertTrue(result.isPresent());
        assertEquals("chargeback_rate", result.get().getName());
        assertEquals(0.008, result.get().getValue().getFloat64Val(), 1e-6);
    }

    @Test
    void testScalarBatch() throws RocksDBException {
        List<RocksDBFeatureStore.ScalarRecord> records = List.of(
                new RocksDBFeatureStore.ScalarRecord("m_001", "gmv_30d", 45230.0),
                new RocksDBFeatureStore.ScalarRecord("m_001", "chargeback_rate", 0.008),
                new RocksDBFeatureStore.ScalarRecord("m_002", "gmv_30d", 12000.0)
        );

        store.putScalarBatch("merchant", records);

        Optional<Feature> f1 = store.getScalarFeature("merchant", "m_001", "gmv_30d");
        assertTrue(f1.isPresent());
        assertEquals(45230.0, f1.get().getValue().getFloat64Val(), 1e-6);

        Optional<Feature> f2 = store.getScalarFeature("merchant", "m_002", "gmv_30d");
        assertTrue(f2.isPresent());
        assertEquals(12000.0, f2.get().getValue().getFloat64Val(), 1e-6);
    }

    @Test
    void testSchema() throws RocksDBException {
        ViewSchema schema = ViewSchema.newBuilder()
                .setViewName("test_view")
                .setVersion(1)
                .addFeatureNames("gmv_30d")
                .addFeatureNames("chargeback_rate")
                .addFeatureDtypes("FLOAT64")
                .addFeatureDtypes("FLOAT64")
                .setSchemaHash(12345)
                .setCreatedAtMs(System.currentTimeMillis())
                .build();

        store.putSchema(schema);

        Optional<ViewSchema> result = store.getSchema("test_view", 1);
        assertTrue(result.isPresent());
        assertEquals("test_view", result.get().getViewName());
        assertEquals(2, result.get().getFeatureNamesCount());
        assertEquals(12345, result.get().getSchemaHash());
    }

    @Test
    void testAssembleVectorFromScalars() throws RocksDBException {
        // Store schema
        ViewSchema schema = ViewSchema.newBuilder()
                .setViewName("assembly_test")
                .setVersion(1)
                .addFeatureNames("gmv_30d")
                .addFeatureNames("chargeback_rate")
                .addFeatureNames("missing_feature")
                .setSchemaHash(999)
                .build();
        store.putSchema(schema);

        // Store some scalars (not all)
        Feature f1 = Feature.newBuilder()
                .setName("gmv_30d")
                .setValue(FeatureValue.newBuilder().setFloat64Val(45230.0).build())
                .setEventTimeMs(System.currentTimeMillis())
                .build();
        Feature f2 = Feature.newBuilder()
                .setName("chargeback_rate")
                .setValue(FeatureValue.newBuilder().setFloat64Val(0.008).build())
                .setEventTimeMs(System.currentTimeMillis())
                .build();
        store.putScalarFeature("merchant", "m_001", f1);
        store.putScalarFeature("merchant", "m_001", f2);

        // Assemble
        Map<String, Double> defaults = Map.of("missing_feature", -1.0);
        FeatureVector vector = store.assembleVectorFromScalars("merchant", "m_001", schema, defaults);

        assertEquals(3, vector.getValuesCount());
        assertEquals(45230.0, vector.getValues(0), 1e-6);
        assertEquals(0.008, vector.getValues(1), 1e-6);
        assertEquals(-1.0, vector.getValues(2), 1e-6); // default

        assertFalse(vector.getIsDefaultMask(0));
        assertFalse(vector.getIsDefaultMask(1));
        assertTrue(vector.getIsDefaultMask(2));
    }

    @Test
    void testEmbeddings() throws RocksDBException {
        float[] embedding = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        store.putEmbedding("merchant", "m_001", "doc_embedding", embedding, 1);

        Optional<float[]> result = store.getEmbedding("merchant", "m_001", "doc_embedding");
        assertTrue(result.isPresent());
        assertEquals(5, result.get().length);
        assertEquals(0.1f, result.get()[0], 1e-6);
        assertEquals(0.5f, result.get()[4], 1e-6);
    }

    @Test
    void testKeyEncoding() {
        byte[] key1 = store.vectorKey("view_a", 1, "entity_1");
        byte[] key2 = store.vectorKey("view_a", 1, "entity_2");
        byte[] key3 = store.vectorKey("view_b", 1, "entity_1");

        // Same view prefix, different entity
        assertArrayEquals(Arrays.copyOf(key1, 6), Arrays.copyOf(key2, 6));
        // Different view prefix
        assertFalse(Arrays.equals(Arrays.copyOf(key1, 4), Arrays.copyOf(key3, 4)));
    }

    @Test
    void testExtractFloat() {
        assertEquals(1.5, RocksDBFeatureStore.extractFloat(
                FeatureValue.newBuilder().setFloat64Val(1.5).build()), 1e-6);
        assertEquals(42.0, RocksDBFeatureStore.extractFloat(
                FeatureValue.newBuilder().setInt64Val(42).build()), 1e-6);
        assertEquals(1.0, RocksDBFeatureStore.extractFloat(
                FeatureValue.newBuilder().setBoolVal(true).build()), 1e-6);
        assertEquals(0.0, RocksDBFeatureStore.extractFloat(
                FeatureValue.newBuilder().setStringVal("text").build()), 1e-6);
    }
}
