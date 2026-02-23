package com.platform.featurestore.proto;

import com.google.protobuf.InvalidProtocolBufferException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Protobuf serialization round-trips to ensure wire format correctness.
 */
class SerializationTest {

    @Test
    void testFeatureVectorRoundTrip() throws InvalidProtocolBufferException {
        FeatureVector original = FeatureVector.newBuilder()
                .setViewName("merchant_fraud_v1")
                .setViewVersion(3)
                .setEntityType("merchant")
                .setEntityId("m_001")
                .addValues(0.008)
                .addValues(45230.0)
                .addValues(423.0)
                .addIsDefaultMask(false)
                .addIsDefaultMask(false)
                .addIsDefaultMask(true)
                .addValueAgesMs(1000L)
                .addValueAgesMs(2000L)
                .addValueAgesMs(-1L)
                .setServedAtMs(System.currentTimeMillis())
                .setRequestId("req-123")
                .setSchemaHash(987654)
                .build();

        byte[] bytes = original.toByteArray();
        FeatureVector parsed = FeatureVector.parseFrom(bytes);

        assertEquals(original.getViewName(), parsed.getViewName());
        assertEquals(original.getViewVersion(), parsed.getViewVersion());
        assertEquals(original.getEntityType(), parsed.getEntityType());
        assertEquals(original.getEntityId(), parsed.getEntityId());
        assertEquals(original.getValuesCount(), parsed.getValuesCount());
        for (int i = 0; i < original.getValuesCount(); i++) {
            assertEquals(original.getValues(i), parsed.getValues(i), 1e-10);
        }
        assertEquals(original.getIsDefaultMaskList(), parsed.getIsDefaultMaskList());
        assertEquals(original.getValueAgesMsList(), parsed.getValueAgesMsList());
        assertEquals(original.getSchemaHash(), parsed.getSchemaHash());
        assertEquals(original.getRequestId(), parsed.getRequestId());
    }

    @Test
    void testFeatureValueVariants() throws InvalidProtocolBufferException {
        // Float64
        FeatureValue f64 = FeatureValue.newBuilder().setFloat64Val(3.14159).build();
        assertEquals(3.14159, FeatureValue.parseFrom(f64.toByteArray()).getFloat64Val(), 1e-10);

        // Float32
        FeatureValue f32 = FeatureValue.newBuilder().setFloat32Val(2.71828f).build();
        assertEquals(2.71828f, FeatureValue.parseFrom(f32.toByteArray()).getFloat32Val(), 1e-5);

        // Int64
        FeatureValue i64 = FeatureValue.newBuilder().setInt64Val(Long.MAX_VALUE).build();
        assertEquals(Long.MAX_VALUE, FeatureValue.parseFrom(i64.toByteArray()).getInt64Val());

        // Bool
        FeatureValue bool = FeatureValue.newBuilder().setBoolVal(true).build();
        assertTrue(FeatureValue.parseFrom(bool.toByteArray()).getBoolVal());

        // String
        FeatureValue str = FeatureValue.newBuilder().setStringVal("high_risk").build();
        assertEquals("high_risk", FeatureValue.parseFrom(str.toByteArray()).getStringVal());
    }

    @Test
    void testFeatureRoundTrip() throws InvalidProtocolBufferException {
        Feature original = Feature.newBuilder()
                .setName("chargeback_rate")
                .setValue(FeatureValue.newBuilder().setFloat64Val(0.008).build())
                .setEventTimeMs(1700000000000L)
                .setCreatedAtMs(1700000001000L)
                .setVersion("v1")
                .setIsDefault(false)
                .build();

        Feature parsed = Feature.parseFrom(original.toByteArray());
        assertEquals(original.getName(), parsed.getName());
        assertEquals(original.getValue().getFloat64Val(), parsed.getValue().getFloat64Val(), 1e-10);
        assertEquals(original.getEventTimeMs(), parsed.getEventTimeMs());
        assertFalse(parsed.getIsDefault());
    }

    @Test
    void testViewSchemaRoundTrip() throws InvalidProtocolBufferException {
        ViewSchema original = ViewSchema.newBuilder()
                .setViewName("fraud_v1")
                .setVersion(2)
                .addFeatureNames("gmv_30d")
                .addFeatureNames("chargeback_rate")
                .addFeatureNames("txn_count")
                .addFeatureDtypes("FLOAT64")
                .addFeatureDtypes("FLOAT64")
                .addFeatureDtypes("INT64")
                .setSchemaHash(12345)
                .setCreatedAtMs(System.currentTimeMillis())
                .build();

        ViewSchema parsed = ViewSchema.parseFrom(original.toByteArray());
        assertEquals(original.getViewName(), parsed.getViewName());
        assertEquals(original.getVersion(), parsed.getVersion());
        assertEquals(original.getFeatureNamesList(), parsed.getFeatureNamesList());
        assertEquals(original.getFeatureDtypesList(), parsed.getFeatureDtypesList());
        assertEquals(original.getSchemaHash(), parsed.getSchemaHash());
    }

    @Test
    void testEmbeddingList() throws InvalidProtocolBufferException {
        Float64List list = Float64List.newBuilder()
                .addValues(0.1).addValues(0.2).addValues(0.3)
                .build();

        FeatureValue fv = FeatureValue.newBuilder()
                .setFloat64ListVal(list)
                .build();

        FeatureValue parsed = FeatureValue.parseFrom(fv.toByteArray());
        assertEquals(3, parsed.getFloat64ListVal().getValuesCount());
        assertEquals(0.2, parsed.getFloat64ListVal().getValues(1), 1e-10);
    }

    @Test
    void testGetFeaturesRequestRoundTrip() throws InvalidProtocolBufferException {
        GetFeaturesRequest original = GetFeaturesRequest.newBuilder()
                .setViewName("fraud_v1")
                .setViewVersion(1)
                .addEntityIds("m_001")
                .addEntityIds("m_002")
                .setIncludeMetadata(true)
                .setRequestId("req-456")
                .build();

        GetFeaturesRequest parsed = GetFeaturesRequest.parseFrom(original.toByteArray());
        assertEquals(2, parsed.getEntityIdsCount());
        assertTrue(parsed.getIncludeMetadata());
    }

    @Test
    void testEmptyVector() throws InvalidProtocolBufferException {
        FeatureVector empty = FeatureVector.newBuilder().build();
        FeatureVector parsed = FeatureVector.parseFrom(empty.toByteArray());
        assertEquals("", parsed.getViewName());
        assertEquals(0, parsed.getValuesCount());
        assertEquals(0, parsed.getSchemaHash());
    }
}
