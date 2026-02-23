package com.platform.featurestore.service;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class FeatureRegistryServiceTest {

    @Test
    void testComputeSchemaHashDeterministic() {
        List<String> names = List.of("gmv_30d", "chargeback_rate", "txn_count");
        int hash1 = FeatureRegistryService.computeSchemaHash(names);
        int hash2 = FeatureRegistryService.computeSchemaHash(names);
        assertEquals(hash1, hash2, "Schema hash must be deterministic");
    }

    @Test
    void testComputeSchemaHashDifferentForDifferentNames() {
        int hash1 = FeatureRegistryService.computeSchemaHash(List.of("gmv_30d", "chargeback_rate"));
        int hash2 = FeatureRegistryService.computeSchemaHash(List.of("gmv_30d", "txn_count"));
        assertNotEquals(hash1, hash2, "Different feature sets must produce different hashes");
    }

    @Test
    void testComputeSchemaHashOrderMatters() {
        int hash1 = FeatureRegistryService.computeSchemaHash(List.of("a", "b", "c"));
        int hash2 = FeatureRegistryService.computeSchemaHash(List.of("c", "b", "a"));
        assertNotEquals(hash1, hash2, "Feature order must affect hash");
    }

    @Test
    void testComputeSchemaHashPositive() {
        int hash = FeatureRegistryService.computeSchemaHash(List.of("feature1", "feature2"));
        assertTrue(hash >= 0, "Schema hash should be non-negative");
    }

    @Test
    void testComputeSchemaHashSingleFeature() {
        int hash = FeatureRegistryService.computeSchemaHash(List.of("only_feature"));
        assertTrue(hash >= 0);
    }
}
