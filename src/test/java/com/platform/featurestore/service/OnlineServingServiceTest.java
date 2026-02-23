package com.platform.featurestore.service;

import com.platform.featurestore.proto.FeatureVector;
import com.platform.featurestore.proto.ViewSchema;
import com.platform.featurestore.store.online.RocksDBFeatureStore;
import com.platform.featurestore.store.online.RedisFeatureCache;
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
class OnlineServingServiceTest {

    @Mock
    private RocksDBFeatureStore rocksStore;

    @Mock
    private RedisFeatureCache redisCache;

    @InjectMocks
    private OnlineServingService servingService;

    private FeatureVector sampleVector(String entityId) {
        return FeatureVector.newBuilder()
                .setViewName("fraud_v1")
                .setViewVersion(1)
                .setEntityType("merchant")
                .setEntityId(entityId)
                .addValues(0.008)
                .addValues(45230.0)
                .setSchemaHash(999)
                .setServedAtMs(System.currentTimeMillis())
                .build();
    }

    @Test
    void testGetFromRedisCache() throws RocksDBException {
        FeatureVector cached = sampleVector("m_001");
        when(redisCache.getVector("fraud_v1", 1, "m_001"))
                .thenReturn(Optional.of(cached));

        OnlineServingService.ServingResult result =
                servingService.getFeatureVector("fraud_v1", 1, "m_001");

        assertNotNull(result.vector());
        assertEquals("REDIS_CACHE", result.source());
        assertEquals("m_001", result.vector().getEntityId());
        verify(rocksStore, never()).getFeatureVector(any(), anyInt(), any());
    }

    @Test
    void testFallbackToRocksDB() throws RocksDBException {
        when(redisCache.getVector("fraud_v1", 1, "m_001"))
                .thenReturn(Optional.empty());
        when(rocksStore.getFeatureVector("fraud_v1", 1, "m_001"))
                .thenReturn(Optional.of(sampleVector("m_001")));

        OnlineServingService.ServingResult result =
                servingService.getFeatureVector("fraud_v1", 1, "m_001");

        assertNotNull(result.vector());
        assertEquals("ROCKSDB_VECTOR", result.source());
        // Should populate cache
        verify(redisCache).putVector(eq("fraud_v1"), eq(1), eq("m_001"), any());
    }

    @Test
    void testFallbackToScalarAssembly() throws RocksDBException {
        when(redisCache.getVector("fraud_v1", 1, "m_001"))
                .thenReturn(Optional.empty());
        when(rocksStore.getFeatureVector("fraud_v1", 1, "m_001"))
                .thenReturn(Optional.empty());

        ViewSchema schema = ViewSchema.newBuilder()
                .setViewName("fraud_v1")
                .setVersion(1)
                .addFeatureNames("f1")
                .setSchemaHash(999)
                .build();
        when(rocksStore.getSchema("fraud_v1", 1)).thenReturn(Optional.of(schema));

        FeatureVector assembled = FeatureVector.newBuilder()
                .setViewName("fraud_v1")
                .addValues(0.0)
                .build();
        when(rocksStore.assembleVectorFromScalars(eq("merchant"), eq("m_001"), eq(schema), any()))
                .thenReturn(assembled);

        OnlineServingService.ServingResult result =
                servingService.getFeatureVector("fraud_v1", 1, "m_001");

        assertNotNull(result.vector());
        assertEquals("ROCKSDB_SCALAR_ASSEMBLY", result.source());
        assertTrue(result.warnings().stream().anyMatch(w -> w.contains("assembled from scalars")));
    }

    @Test
    void testCompleteMiss() throws RocksDBException {
        when(redisCache.getVector("fraud_v1", 1, "m_001"))
                .thenReturn(Optional.empty());
        when(rocksStore.getFeatureVector("fraud_v1", 1, "m_001"))
                .thenReturn(Optional.empty());
        when(rocksStore.getSchema("fraud_v1", 1)).thenReturn(Optional.empty());

        OnlineServingService.ServingResult result =
                servingService.getFeatureVector("fraud_v1", 1, "m_001");

        assertNull(result.vector());
        assertEquals("MISS", result.source());
    }

    @Test
    void testGetWithNoRedisCache() throws RocksDBException {
        // Construct service with null redis
        OnlineServingService noRedisService = new OnlineServingService(rocksStore, null);

        when(rocksStore.getFeatureVector("fraud_v1", 1, "m_001"))
                .thenReturn(Optional.of(sampleVector("m_001")));

        OnlineServingService.ServingResult result =
                noRedisService.getFeatureVector("fraud_v1", 1, "m_001");

        assertNotNull(result.vector());
        assertEquals("ROCKSDB_VECTOR", result.source());
    }

    @Test
    void testBatchServing() throws RocksDBException {
        List<String> ids = List.of("m_001", "m_002", "m_003");

        when(redisCache.getVectorsBatch("fraud_v1", 1, ids))
                .thenReturn(Map.of("m_001", sampleVector("m_001")));
        when(rocksStore.getFeatureVectorsBatch(eq("fraud_v1"), eq(1), eq(List.of("m_002", "m_003"))))
                .thenReturn(Map.of("m_002", sampleVector("m_002")));

        OnlineServingService.BatchServingResult result =
                servingService.getFeatureVectorsBatch("fraud_v1", 1, ids);

        assertEquals(2, result.vectors().size()); // m_001 from cache, m_002 from rocks, m_003 missing
        assertTrue(result.warnings().stream().anyMatch(w -> w.contains("Missing vectors for 1")));
    }
}
