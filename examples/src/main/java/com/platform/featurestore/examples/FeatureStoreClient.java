package com.platform.featurestore.examples;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.platform.featurestore.proto.*;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

import java.io.Closeable;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * Standalone Java client for the ML Feature Store.
 * <p>
 * Uses {@code java.net.http.HttpClient} for REST (registry, materialization, offline)
 * and gRPC stubs for online serving and scalar writes.
 */
public class FeatureStoreClient implements Closeable {

    private final String restBaseUrl;
    private final HttpClient http;
    private final ObjectMapper json;
    private final ManagedChannel grpcChannel;
    private final FeatureStoreServiceGrpc.FeatureStoreServiceBlockingStub grpcStub;

    public FeatureStoreClient(String restBaseUrl, String grpcTarget) {
        this.restBaseUrl = restBaseUrl;
        this.http = HttpClient.newHttpClient();
        this.json = new ObjectMapper();

        String[] parts = grpcTarget.split(":");
        this.grpcChannel = ManagedChannelBuilder
                .forAddress(parts[0], Integer.parseInt(parts[1]))
                .usePlaintext()
                .build();
        this.grpcStub = FeatureStoreServiceGrpc.newBlockingStub(grpcChannel);
    }

    // -----------------------------------------------------------------------
    // Registry (REST)
    // -----------------------------------------------------------------------

    /** Create an entity and return its UUID. */
    public String createEntity(String name, String description, String joinKey, String joinKeyType)
            throws IOException, InterruptedException {
        Map<String, Object> body = Map.of(
                "name", name, "description", description,
                "joinKey", joinKey, "joinKeyType", joinKeyType);
        JsonNode resp = postJson("/api/v1/entities", body);
        return resp.get("id").asText();
    }

    /** Look up an entity by name. Returns UUID or null. */
    public String getEntityByName(String name) throws IOException, InterruptedException {
        JsonNode resp = getJson("/api/v1/entities/by-name/" + name);
        return resp != null && resp.has("id") ? resp.get("id").asText() : null;
    }

    /** Create a feature and return its UUID. */
    public String createFeature(String name, String entityId, String dtype, String description,
                                String owner, String sourcePipeline, String updateFrequency,
                                int maxAgeSeconds, String defaultValue)
            throws IOException, InterruptedException {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("name", name);
        body.put("entityId", entityId);
        body.put("dtype", dtype);
        body.put("description", description);
        body.put("owner", owner);
        body.put("sourcePipeline", sourcePipeline);
        body.put("updateFrequency", updateFrequency);
        body.put("maxAgeSeconds", maxAgeSeconds);
        body.put("defaultValue", defaultValue);
        JsonNode resp = postJson("/api/v1/features", body);
        return resp.get("id").asText();
    }

    /** List features for an entity. Returns list of {id, name, ...} maps. */
    public List<Map<String, Object>> listFeatures(String entityId)
            throws IOException, InterruptedException {
        JsonNode resp = getJson("/api/v1/features?entityId=" + entityId);
        if (resp == null || !resp.isArray()) return List.of();
        return json.convertValue(resp, new TypeReference<>() {});
    }

    /** Create a feature view and return its UUID. */
    public String createFeatureView(String name, int version, String entityId,
                                    String description, String modelName,
                                    String mlFramework, List<String> featureIds)
            throws IOException, InterruptedException {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("name", name);
        body.put("version", version);
        body.put("entityId", entityId);
        body.put("description", description);
        body.put("modelName", modelName);
        body.put("mlFramework", mlFramework);
        body.put("featureIds", featureIds);
        JsonNode resp = postJson("/api/v1/feature-views", body);
        return resp.get("id").asText();
    }

    // -----------------------------------------------------------------------
    // Materialization (REST)
    // -----------------------------------------------------------------------

    /** Materialize a single feature vector via REST. */
    public void materializeVector(String viewName, int viewVersion, String entityType,
                                  String entityId, List<Double> values, int schemaHash)
            throws IOException, InterruptedException {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("viewName", viewName);
        body.put("viewVersion", viewVersion);
        body.put("entityType", entityType);
        body.put("entityId", entityId);
        body.put("values", values);
        body.put("schemaHash", schemaHash);
        postJson("/api/v1/materialize/vector", body);
    }

    /** Materialize a batch of feature vectors via REST. Returns count materialized. */
    public int materializeVectorBatch(List<Map<String, Object>> vectors)
            throws IOException, InterruptedException {
        Map<String, Object> body = Map.of("vectors", vectors);
        JsonNode resp = postJson("/api/v1/materialize/batch", body);
        return resp.has("count") ? resp.get("count").asInt() : 0;
    }

    // -----------------------------------------------------------------------
    // Scalar writes (gRPC — individual feature updates)
    // -----------------------------------------------------------------------

    /** Write individual scalar features for an entity via gRPC. */
    public int putScalarFeatures(String entityType, String entityId,
                                 Map<String, Double> featureValues) {
        List<Feature> features = new ArrayList<>();
        for (var entry : featureValues.entrySet()) {
            features.add(Feature.newBuilder()
                    .setName(entry.getKey())
                    .setValue(FeatureValue.newBuilder()
                            .setFloat64Val(entry.getValue())
                            .build())
                    .setEventTimeMs(System.currentTimeMillis())
                    .build());
        }

        PutScalarFeaturesRequest request = PutScalarFeaturesRequest.newBuilder()
                .setEntityType(entityType)
                .setEntityId(entityId)
                .addAllFeatures(features)
                .build();

        PutScalarFeaturesResponse response = grpcStub.putScalarFeatures(request);
        return response.getFeaturesWritten();
    }

    // -----------------------------------------------------------------------
    // Online serving (gRPC)
    // -----------------------------------------------------------------------

    /** Fetch feature vectors for a batch of entity IDs. */
    public GetFeaturesResponse getOnlineFeatures(String viewName, int viewVersion,
                                                  List<String> entityIds) {
        GetFeaturesRequest request = GetFeaturesRequest.newBuilder()
                .setViewName(viewName)
                .setViewVersion(viewVersion)
                .addAllEntityIds(entityIds)
                .build();
        return grpcStub.getOnlineFeatures(request);
    }

    /** Get the schema for a feature view. */
    public ViewSchema getViewSchema(String viewName, int version) {
        GetViewSchemaRequest request = GetViewSchemaRequest.newBuilder()
                .setViewName(viewName)
                .setVersion(version)
                .build();
        return grpcStub.getViewSchema(request).getSchema();
    }

    // -----------------------------------------------------------------------
    // Offline store (REST)
    // -----------------------------------------------------------------------

    /** Write feature records to the Iceberg offline store (attribute form). */
    public int writeOfflineRecords(List<Map<String, Object>> records)
            throws IOException, InterruptedException {
        Map<String, Object> body = Map.of("records", records);
        JsonNode resp = postJson("/api/v1/offline/write-records", body);
        return resp.has("count") ? resp.get("count").asInt() : 0;
    }

    /** Generate ASOF-joined training data from the offline store. */
    public List<Map<String, Object>> getTrainingData(String entityType,
                                                       List<String> featureNames,
                                                       List<Map<String, Object>> labelEvents)
            throws IOException, InterruptedException {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("entityType", entityType);
        body.put("featureNames", featureNames);
        body.put("labelEvents", labelEvents);
        JsonNode resp = postJson("/api/v1/offline/training-data", body);
        if (resp.has("rows")) {
            return json.convertValue(resp.get("rows"), new TypeReference<>() {});
        }
        return List.of();
    }

    // -----------------------------------------------------------------------
    // HTTP helpers
    // -----------------------------------------------------------------------

    private JsonNode postJson(String path, Object body) throws IOException, InterruptedException {
        String jsonBody = json.writeValueAsString(body);
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(restBaseUrl + path))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();
        HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() >= 400) {
            throw new IOException("HTTP " + response.statusCode() + ": " + response.body());
        }
        return json.readTree(response.body());
    }

    private JsonNode getJson(String path) throws IOException, InterruptedException {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(restBaseUrl + path))
                .GET()
                .build();
        HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() == 404) return null;
        if (response.statusCode() >= 400) {
            throw new IOException("HTTP " + response.statusCode() + ": " + response.body());
        }
        return json.readTree(response.body());
    }

    @Override
    public void close() {
        try {
            grpcChannel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            grpcChannel.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}
