package com.platform.featurestore.service;

import com.platform.featurestore.proto.FeatureVector;
import com.platform.featurestore.proto.ViewSchema;
import com.platform.featurestore.store.online.RocksDBFeatureStore;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.rocksdb.RocksDBException;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class ValidationServiceTest {

    @Mock
    private RocksDBFeatureStore rocksStore;

    @InjectMocks
    private ValidationService validationService;

    // --- Schema hash validation ---

    @Test
    void testSchemaHashMatch() {
        ViewSchema schema = ViewSchema.newBuilder()
                .setViewName("test_view")
                .setVersion(1)
                .addFeatureNames("gmv_30d")
                .addFeatureNames("chargeback_rate")
                .setSchemaHash(12345)
                .build();

        FeatureVector vector = FeatureVector.newBuilder()
                .setSchemaHash(12345)
                .addValues(1.0)
                .addValues(2.0)
                .build();

        ValidationService.ValidationResult result = validationService.validateSchemaHash(vector, schema);
        assertTrue(result.passed());
        assertEquals(2, result.gates().size());
        assertTrue(result.gates().get(0).passed()); // hash match
        assertTrue(result.gates().get(1).passed()); // length match
    }

    @Test
    void testSchemaHashMismatch() {
        ViewSchema schema = ViewSchema.newBuilder()
                .setSchemaHash(12345)
                .addFeatureNames("a")
                .build();

        FeatureVector vector = FeatureVector.newBuilder()
                .setSchemaHash(99999)
                .addValues(1.0)
                .build();

        ValidationService.ValidationResult result = validationService.validateSchemaHash(vector, schema);
        assertFalse(result.gates().get(0).passed());
        assertTrue(result.gates().get(0).message().contains("Schema mismatch"));
    }

    @Test
    void testVectorLengthMismatch() {
        ViewSchema schema = ViewSchema.newBuilder()
                .setSchemaHash(12345)
                .addFeatureNames("a")
                .addFeatureNames("b")
                .addFeatureNames("c")
                .build();

        FeatureVector vector = FeatureVector.newBuilder()
                .setSchemaHash(12345)
                .addValues(1.0) // only 1 value, schema expects 3
                .build();

        ValidationService.ValidationResult result = validationService.validateSchemaHash(vector, schema);
        assertFalse(result.gates().get(1).passed()); // length mismatch
    }

    // --- Staleness validation ---

    @Test
    void testStalenessWithinSla() {
        long nowMs = System.currentTimeMillis();
        FeatureVector vector = FeatureVector.newBuilder()
                .addValues(1.0)
                .addValues(2.0)
                .addValueAgesMs(5000L)  // 5 seconds old
                .addValueAgesMs(3000L)  // 3 seconds old
                .addIsDefaultMask(false)
                .addIsDefaultMask(false)
                .build();

        Map<String, Integer> maxAge = Map.of(
                "gmv_30d", 3600, // 1 hour
                "chargeback_rate", 3600
        );
        List<String> names = List.of("gmv_30d", "chargeback_rate");

        ValidationService.ValidationResult result = validationService.validateStaleness(vector, maxAge, names);
        assertTrue(result.passed());
    }

    @Test
    void testStalenessViolation() {
        FeatureVector vector = FeatureVector.newBuilder()
                .addValues(1.0)
                .addValueAgesMs(7200_000L) // 2 hours
                .addIsDefaultMask(false)
                .build();

        Map<String, Integer> maxAge = Map.of("gmv_30d", 3600); // 1 hour SLA
        List<String> names = List.of("gmv_30d");

        ValidationService.ValidationResult result = validationService.validateStaleness(vector, maxAge, names);
        assertFalse(result.passed());
        assertTrue(result.gates().get(0).message().contains("STALE"));
    }

    @Test
    void testStalenessWithDefaults() {
        FeatureVector vector = FeatureVector.newBuilder()
                .addValues(0.0)
                .addValues(1.0)
                .addValueAgesMs(-1L) // unknown age
                .addValueAgesMs(5000L)
                .addIsDefaultMask(true)
                .addIsDefaultMask(false)
                .build();

        Map<String, Integer> maxAge = Map.of("feature_b", 3600);
        List<String> names = List.of("feature_a", "feature_b");

        ValidationService.ValidationResult result = validationService.validateStaleness(vector, maxAge, names);
        assertFalse(result.warnings().isEmpty());
        assertTrue(result.warnings().stream().anyMatch(w -> w.contains("default-filled")));
    }

    // --- Model metrics validation ---

    @Test
    void testModelMetricsPassing() {
        ValidationService.ValidationResult result = validationService.validateModelMetrics(
                0.85, 0.55, 0.10, null);
        assertTrue(result.passed());
        assertEquals(3, result.gates().size());
    }

    @Test
    void testModelMetricsFailingAucRoc() {
        ValidationService.ValidationResult result = validationService.validateModelMetrics(
                0.60, 0.55, 0.10, null);
        assertFalse(result.passed());
        assertFalse(result.gates().get(0).passed()); // AUC-ROC < 0.75
    }

    @Test
    void testModelMetricsFailingAucPr() {
        ValidationService.ValidationResult result = validationService.validateModelMetrics(
                0.85, 0.30, 0.10, null);
        assertFalse(result.passed());
        assertFalse(result.gates().get(1).passed()); // AUC-PR < 0.40
    }

    @Test
    void testModelMetricsLowScoreStd() {
        ValidationService.ValidationResult result = validationService.validateModelMetrics(
                0.85, 0.55, 0.01, null);
        assertFalse(result.passed());
        assertFalse(result.gates().get(2).passed()); // score std < 0.05
    }

    @Test
    void testModelMetricsWithBaselineRegression() {
        ValidationService.ValidationResult result = validationService.validateModelMetrics(
                0.80, 0.55, 0.10, 0.85);
        // Degradation = 0.85 - 0.80 = 0.05 > 0.02 threshold
        assertFalse(result.passed());
        assertEquals(4, result.gates().size());
        assertFalse(result.gates().get(3).passed());
    }

    @Test
    void testModelMetricsBaselineNoRegression() {
        ValidationService.ValidationResult result = validationService.validateModelMetrics(
                0.84, 0.55, 0.10, 0.85);
        // Degradation = 0.85 - 0.84 = 0.01 < 0.02 threshold
        assertTrue(result.passed());
        assertTrue(result.gates().get(3).passed());
    }

    // --- Feature vector validation ---

    @Test
    void testValidateFeatureVectorSchemaNotFound() throws RocksDBException {
        when(rocksStore.getSchema("view_x", 1)).thenReturn(Optional.empty());

        ValidationService.ValidationResult result =
                validationService.validateFeatureVector("view_x", 1, "m_001");
        assertFalse(result.passed());
        assertEquals("VALIDATION_ERROR", result.type());
    }

    @Test
    void testValidateFeatureVectorNotFound() throws RocksDBException {
        ViewSchema schema = ViewSchema.newBuilder()
                .setViewName("view_x")
                .setVersion(1)
                .setSchemaHash(999)
                .build();
        when(rocksStore.getSchema("view_x", 1)).thenReturn(Optional.of(schema));
        when(rocksStore.getFeatureVector("view_x", 1, "m_001")).thenReturn(Optional.empty());

        ValidationService.ValidationResult result =
                validationService.validateFeatureVector("view_x", 1, "m_001");
        assertFalse(result.passed());
    }

    @Test
    void testValidateFeatureVectorSuccess() throws RocksDBException {
        ViewSchema schema = ViewSchema.newBuilder()
                .setViewName("view_x")
                .setVersion(1)
                .addFeatureNames("f1")
                .addFeatureNames("f2")
                .setSchemaHash(999)
                .build();
        FeatureVector vector = FeatureVector.newBuilder()
                .setSchemaHash(999)
                .addValues(1.0)
                .addValues(2.0)
                .build();

        when(rocksStore.getSchema("view_x", 1)).thenReturn(Optional.of(schema));
        when(rocksStore.getFeatureVector("view_x", 1, "m_001")).thenReturn(Optional.of(vector));

        ValidationService.ValidationResult result =
                validationService.validateFeatureVector("view_x", 1, "m_001");
        assertTrue(result.passed());
    }
}
