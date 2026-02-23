package com.platform.featurestore.dto;

import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * REST API DTOs. Using records for conciseness.
 */
public final class Dtos {

    private Dtos() {}

    // --- Entity DTOs ---

    public record CreateEntityRequest(
            String name, String description, String joinKey, String joinKeyType) {}

    public record EntityResponse(
            UUID id, String name, String description, String joinKey, String joinKeyType) {}

    // --- Feature DTOs ---

    public record CreateFeatureRequest(
            String name, UUID entityId, String dtype, String description,
            String owner, String sourcePipeline, String updateFrequency,
            Integer maxAgeSeconds, String defaultValue) {}

    public record FeatureResponse(
            UUID id, String name, UUID entityId, String dtype, String description,
            String owner, String updateFrequency, Integer maxAgeSeconds,
            String status, int version) {}

    // --- Feature View DTOs ---

    public record CreateFeatureViewRequest(
            String name, int version, UUID entityId, String description,
            String modelName, String mlFramework, List<UUID> featureIds) {}

    public record FeatureViewResponse(
            UUID id, String name, int version, UUID entityId, String description,
            String modelName, int vectorLength, int schemaHash, String status,
            List<String> featureNames) {}

    // --- Serving DTOs ---

    public record GetFeaturesRestRequest(
            String viewName, int viewVersion, List<String> entityIds) {}

    public record FeatureVectorResponse(
            String viewName, int viewVersion, String entityType, String entityId,
            List<Double> values, int schemaHash, long servedAtMs) {}

    public record ServingResponse(
            List<FeatureVectorResponse> vectors, long latencyUs, List<String> warnings) {}

    // --- Materialization DTOs ---

    public record MaterializeVectorRequest(
            String viewName, int viewVersion, String entityType, String entityId,
            List<Double> values, int schemaHash) {}

    public record MaterializeBatchRequest(
            List<MaterializeVectorRequest> vectors) {}

    public record MaterializeResponse(
            boolean success, int count, String message) {}

    // --- Validation DTOs ---

    public record ValidationGateResponse(
            String name, boolean passed, double metric, double threshold, String message) {}

    public record ValidationResponse(
            String type, boolean passed, List<ValidationGateResponse> gates, List<String> warnings) {}

    public record ValidateModelRequest(
            double aucRoc, double aucPr, double scoreStd, Double baselineAucRoc) {}

    // --- Offline Store DTOs ---

    public record OfflineFeatureRecord(
            String entityType, String entityId, String featureName,
            Double valueFloat, String valueString, String eventTime,
            String pipelineId, Integer viewVersion) {}

    public record WriteOfflineRecordsRequest(List<OfflineFeatureRecord> records) {}

    public record WriteOfflineRecordsResponse(boolean success, int count, String message) {}

    public record LabelEventDto(String entityId, String eventTime, int label) {}

    public record TrainingDataRequest(
            String entityType, List<String> featureNames,
            List<LabelEventDto> labelEvents) {}

    public record TrainingDataResponse(
            List<Map<String, Object>> rows, int rowCount, int featureCount) {}
}
