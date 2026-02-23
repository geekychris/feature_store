package com.platform.featurestore.controller;

import com.platform.featurestore.dto.Dtos.*;
import com.platform.featurestore.proto.FeatureVector;
import com.platform.featurestore.service.MaterializationService;
import com.platform.featurestore.service.OnlineServingService;
import com.platform.featurestore.service.ValidationService;
import org.rocksdb.RocksDBException;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;

@RestController
@RequestMapping("/api/v1")
public class ServingController {

    private final OnlineServingService servingService;
    private final MaterializationService materializationService;
    private final ValidationService validationService;

    public ServingController(OnlineServingService servingService,
                              MaterializationService materializationService,
                              ValidationService validationService) {
        this.servingService = servingService;
        this.materializationService = materializationService;
        this.validationService = validationService;
    }

    // --- Online Serving ---

    @PostMapping("/serving/online")
    public ResponseEntity<ServingResponse> getOnlineFeatures(@RequestBody GetFeaturesRestRequest req) {
        OnlineServingService.BatchServingResult result = servingService.getFeatureVectorsBatch(
                req.viewName(), req.viewVersion(), req.entityIds());

        List<FeatureVectorResponse> vectorResponses = result.vectors().stream()
                .map(this::toVectorResponse)
                .toList();

        return ResponseEntity.ok(new ServingResponse(
                vectorResponses, result.latencyUs(), result.warnings()));
    }

    @GetMapping("/serving/online/{viewName}/{viewVersion}/{entityId}")
    public ResponseEntity<FeatureVectorResponse> getFeature(
            @PathVariable String viewName,
            @PathVariable int viewVersion,
            @PathVariable String entityId) {

        OnlineServingService.ServingResult result = servingService.getFeatureVector(
                viewName, viewVersion, entityId);

        if (result.vector() == null) {
            return ResponseEntity.notFound().build();
        }

        return ResponseEntity.ok(toVectorResponse(result.vector()));
    }

    // --- Materialization ---

    @PostMapping("/materialize/vector")
    public ResponseEntity<MaterializeResponse> materializeVector(@RequestBody MaterializeVectorRequest req) {
        try {
            FeatureVector vector = buildVector(req);
            materializationService.materializeVector(vector);
            return ResponseEntity.ok(new MaterializeResponse(true, 1, "Materialized"));
        } catch (RocksDBException e) {
            return ResponseEntity.internalServerError()
                    .body(new MaterializeResponse(false, 0, e.getMessage()));
        }
    }

    @PostMapping("/materialize/batch")
    public ResponseEntity<MaterializeResponse> materializeBatch(@RequestBody MaterializeBatchRequest req) {
        try {
            List<FeatureVector> vectors = req.vectors().stream()
                    .map(this::buildVector)
                    .toList();
            int count = materializationService.materializeVectorBatch(vectors);
            return ResponseEntity.ok(new MaterializeResponse(true, count, "Batch materialized"));
        } catch (RocksDBException e) {
            return ResponseEntity.internalServerError()
                    .body(new MaterializeResponse(false, 0, e.getMessage()));
        }
    }

    // --- Validation ---

    @GetMapping("/validate/{viewName}/{viewVersion}/{entityId}")
    public ResponseEntity<ValidationResponse> validateVector(
            @PathVariable String viewName,
            @PathVariable int viewVersion,
            @PathVariable String entityId) {

        ValidationService.ValidationResult result =
                validationService.validateFeatureVector(viewName, viewVersion, entityId);
        return ResponseEntity.ok(toValidationResponse(result));
    }

    @PostMapping("/validate/model")
    public ResponseEntity<ValidationResponse> validateModel(@RequestBody ValidateModelRequest req) {
        ValidationService.ValidationResult result = validationService.validateModelMetrics(
                req.aucRoc(), req.aucPr(), req.scoreStd(), req.baselineAucRoc());
        return ResponseEntity.ok(toValidationResponse(result));
    }

    // --- Mappers ---

    private FeatureVectorResponse toVectorResponse(FeatureVector fv) {
        return new FeatureVectorResponse(
                fv.getViewName(), fv.getViewVersion(),
                fv.getEntityType(), fv.getEntityId(),
                new ArrayList<>(fv.getValuesList()),
                fv.getSchemaHash(), fv.getServedAtMs());
    }

    private FeatureVector buildVector(MaterializeVectorRequest req) {
        return FeatureVector.newBuilder()
                .setViewName(req.viewName())
                .setViewVersion(req.viewVersion())
                .setEntityType(req.entityType())
                .setEntityId(req.entityId())
                .addAllValues(req.values())
                .setSchemaHash(req.schemaHash())
                .setServedAtMs(System.currentTimeMillis())
                .build();
    }

    private ValidationResponse toValidationResponse(ValidationService.ValidationResult result) {
        List<ValidationGateResponse> gates = result.gates().stream()
                .map(g -> new ValidationGateResponse(
                        g.name(), g.passed(), g.metric(), g.threshold(), g.message()))
                .toList();
        return new ValidationResponse(result.type(), result.passed(), gates, result.warnings());
    }
}
