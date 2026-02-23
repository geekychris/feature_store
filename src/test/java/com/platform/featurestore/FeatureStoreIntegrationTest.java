package com.platform.featurestore;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.platform.featurestore.dto.Dtos.*;
import org.junit.jupiter.api.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.util.List;
import java.util.UUID;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/**
 * Full integration test with Testcontainers PostgreSQL.
 * Tests the REST API → Service → JPA → PostgreSQL path.
 * Redis is disabled; RocksDB and Iceberg use temp directories.
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@AutoConfigureMockMvc
@Testcontainers
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class FeatureStoreIntegrationTest {

    @Container
    static PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:16-alpine")
            .withDatabaseName("feature_store_test")
            .withUsername("test")
            .withPassword("test");

    @DynamicPropertySource
    static void configureProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", postgres::getJdbcUrl);
        registry.add("spring.datasource.username", postgres::getUsername);
        registry.add("spring.datasource.password", postgres::getPassword);
        registry.add("spring.datasource.driver-class-name", () -> "org.postgresql.Driver");
        registry.add("spring.jpa.hibernate.ddl-auto", () -> "validate");
        registry.add("spring.flyway.enabled", () -> "true");
        // Disable Redis
        registry.add("feature-store.redis.enabled", () -> "false");
        registry.add("spring.data.redis.host", () -> "localhost");
        registry.add("spring.data.redis.port", () -> "16379"); // unlikely port
        registry.add("spring.autoconfigure.exclude", () ->
                "org.springframework.boot.autoconfigure.data.redis.RedisAutoConfiguration," +
                "org.springframework.boot.autoconfigure.data.redis.RedisRepositoriesAutoConfiguration");
        // Temp paths for RocksDB and Iceberg
        registry.add("feature-store.rocksdb.path", () -> "/tmp/feature-store-test-" + System.nanoTime() + "/rocksdb");
        registry.add("feature-store.iceberg.warehouse", () -> "/tmp/feature-store-test-" + System.nanoTime() + "/iceberg");
        // Disable gRPC server for tests
        registry.add("grpc.server.port", () -> "-1");
    }

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    private static UUID entityId;
    private static UUID featureId1;
    private static UUID featureId2;

    @Test
    @Order(1)
    void testCreateEntity() throws Exception {
        CreateEntityRequest req = new CreateEntityRequest(
                "merchant", "Test merchant entity", "merchant_id", "STRING");

        MvcResult result = mockMvc.perform(post("/api/v1/entities")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.name").value("merchant"))
                .andExpect(jsonPath("$.joinKey").value("merchant_id"))
                .andReturn();

        EntityResponse response = objectMapper.readValue(
                result.getResponse().getContentAsString(), EntityResponse.class);
        entityId = response.id();
    }

    @Test
    @Order(2)
    void testListEntities() throws Exception {
        mockMvc.perform(get("/api/v1/entities"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].name").value("merchant"));
    }

    @Test
    @Order(3)
    void testGetEntityByName() throws Exception {
        mockMvc.perform(get("/api/v1/entities/by-name/merchant"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.name").value("merchant"));
    }

    @Test
    @Order(4)
    void testGetEntityNotFound() throws Exception {
        mockMvc.perform(get("/api/v1/entities/" + UUID.randomUUID()))
                .andExpect(status().isNotFound());
    }

    @Test
    @Order(5)
    void testCreateFeatures() throws Exception {
        // Feature 1: gmv_30d
        CreateFeatureRequest req1 = new CreateFeatureRequest(
                "gmv_30d", entityId, "FLOAT64", "Gross merchandise value last 30 days",
                "risk-team", "daily_pipeline", "DAILY", 86400, "0.0");

        MvcResult result1 = mockMvc.perform(post("/api/v1/features")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req1)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.name").value("gmv_30d"))
                .andExpect(jsonPath("$.status").value("ACTIVE"))
                .andReturn();

        FeatureResponse resp1 = objectMapper.readValue(
                result1.getResponse().getContentAsString(), FeatureResponse.class);
        featureId1 = resp1.id();

        // Feature 2: chargeback_rate
        CreateFeatureRequest req2 = new CreateFeatureRequest(
                "chargeback_rate", entityId, "FLOAT64", "Chargeback rate",
                "risk-team", "daily_pipeline", "DAILY", 86400, "0.0");

        MvcResult result2 = mockMvc.perform(post("/api/v1/features")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req2)))
                .andExpect(status().isCreated())
                .andReturn();

        FeatureResponse resp2 = objectMapper.readValue(
                result2.getResponse().getContentAsString(), FeatureResponse.class);
        featureId2 = resp2.id();
    }

    @Test
    @Order(6)
    void testListFeaturesByEntity() throws Exception {
        mockMvc.perform(get("/api/v1/features").param("entityId", entityId.toString()))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(2));
    }

    @Test
    @Order(7)
    void testCreateFeatureView() throws Exception {
        CreateFeatureViewRequest req = new CreateFeatureViewRequest(
                "merchant_fraud", 1, entityId,
                "Merchant fraud detection features",
                "fraud_xgb_v1", "XGBOOST",
                List.of(featureId1, featureId2));

        mockMvc.perform(post("/api/v1/feature-views")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.name").value("merchant_fraud"))
                .andExpect(jsonPath("$.version").value(1))
                .andExpect(jsonPath("$.vectorLength").value(2))
                .andExpect(jsonPath("$.featureNames[0]").value("gmv_30d"))
                .andExpect(jsonPath("$.featureNames[1]").value("chargeback_rate"));
    }

    @Test
    @Order(8)
    void testGetFeatureViewByNameAndVersion() throws Exception {
        mockMvc.perform(get("/api/v1/feature-views/merchant_fraud/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.name").value("merchant_fraud"))
                .andExpect(jsonPath("$.schemaHash").isNumber());
    }

    @Test
    @Order(9)
    void testGetLatestFeatureView() throws Exception {
        mockMvc.perform(get("/api/v1/feature-views/merchant_fraud/latest"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.version").value(1));
    }

    @Test
    @Order(10)
    void testMaterializeAndServe() throws Exception {
        // Materialize a vector
        MaterializeVectorRequest materializeReq = new MaterializeVectorRequest(
                "merchant_fraud", 1, "merchant", "m_test_001",
                List.of(45230.0, 0.008), 0);

        mockMvc.perform(post("/api/v1/materialize/vector")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(materializeReq)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true));

        // Serve the vector
        mockMvc.perform(get("/api/v1/serving/online/merchant_fraud/1/m_test_001"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.entityId").value("m_test_001"))
                .andExpect(jsonPath("$.values[0]").value(45230.0))
                .andExpect(jsonPath("$.values[1]").value(0.008));
    }

    @Test
    @Order(11)
    void testBatchMaterializeAndServe() throws Exception {
        MaterializeVectorRequest v1 = new MaterializeVectorRequest(
                "merchant_fraud", 1, "merchant", "m_batch_001",
                List.of(10000.0, 0.001), 0);
        MaterializeVectorRequest v2 = new MaterializeVectorRequest(
                "merchant_fraud", 1, "merchant", "m_batch_002",
                List.of(20000.0, 0.002), 0);

        MaterializeBatchRequest batchReq = new MaterializeBatchRequest(List.of(v1, v2));

        mockMvc.perform(post("/api/v1/materialize/batch")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(batchReq)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true))
                .andExpect(jsonPath("$.count").value(2));

        // Batch serve
        GetFeaturesRestRequest serveReq = new GetFeaturesRestRequest(
                "merchant_fraud", 1, List.of("m_batch_001", "m_batch_002"));

        mockMvc.perform(post("/api/v1/serving/online")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(serveReq)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.vectors.length()").value(2));
    }

    @Test
    @Order(12)
    void testValidateModel() throws Exception {
        ValidateModelRequest req = new ValidateModelRequest(0.85, 0.55, 0.10, null);

        mockMvc.perform(post("/api/v1/validate/model")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.passed").value(true))
                .andExpect(jsonPath("$.type").value("MODEL_VALIDATION"));
    }

    @Test
    @Order(13)
    void testValidateModelFailing() throws Exception {
        ValidateModelRequest req = new ValidateModelRequest(0.60, 0.30, 0.01, null);

        mockMvc.perform(post("/api/v1/validate/model")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.passed").value(false));
    }

    @Test
    @Order(14)
    void testDeprecateFeature() throws Exception {
        mockMvc.perform(post("/api/v1/features/" + featureId2 + "/deprecate")
                        .param("message", "Replaced by chargeback_rate_v2"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("DEPRECATED"));
    }
}
