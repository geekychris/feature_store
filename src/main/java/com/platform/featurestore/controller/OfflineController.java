package com.platform.featurestore.controller;

import com.platform.featurestore.dto.Dtos.*;
import com.platform.featurestore.service.MaterializationService;
import com.platform.featurestore.store.offline.IcebergOfflineStore;
import com.platform.featurestore.store.offline.IcebergOfflineStore.FeatureRecordData;
import com.platform.featurestore.store.offline.IcebergOfflineStore.LabelEvent;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/offline")
public class OfflineController {

    private final MaterializationService materializationService;
    private final IcebergOfflineStore offlineStore;

    public OfflineController(MaterializationService materializationService,
                             IcebergOfflineStore offlineStore) {
        this.materializationService = materializationService;
        this.offlineStore = offlineStore;
    }

    /**
     * Write feature records to the Iceberg offline store in attribute form
     * (entity_id, feature_name, value per row).
     */
    @PostMapping("/write-records")
    public ResponseEntity<WriteOfflineRecordsResponse> writeRecords(
            @RequestBody WriteOfflineRecordsRequest request) {
        try {
            OffsetDateTime now = OffsetDateTime.now();
            List<FeatureRecordData> records = request.records().stream()
                    .map(r -> new FeatureRecordData(
                            r.entityType(),
                            r.entityId(),
                            r.featureName(),
                            r.valueFloat(),
                            r.valueString(),
                            r.eventTime() != null ? OffsetDateTime.parse(r.eventTime()) : now,
                            now,
                            r.pipelineId(),
                            r.viewVersion()
                    ))
                    .toList();

            offlineStore.writeFeatureRecords(records);
            return ResponseEntity.ok(new WriteOfflineRecordsResponse(
                    true, records.size(), "Wrote " + records.size() + " records to offline store"));
        } catch (Exception e) {
            return ResponseEntity.internalServerError()
                    .body(new WriteOfflineRecordsResponse(false, 0, e.getMessage()));
        }
    }

    /**
     * Generate point-in-time correct training data via ASOF join.
     * For each label event, finds the latest feature values before the event time.
     */
    @PostMapping("/training-data")
    public ResponseEntity<TrainingDataResponse> getTrainingData(
            @RequestBody TrainingDataRequest request) {
        try {
            List<LabelEvent> labelEvents = request.labelEvents().stream()
                    .map(le -> new LabelEvent(
                            le.entityId(),
                            le.eventTime() != null
                                    ? OffsetDateTime.parse(le.eventTime())
                                    : OffsetDateTime.now(),
                            le.label()
                    ))
                    .toList();

            List<Map<String, Object>> rows = offlineStore.generateTrainingDataset(
                    request.entityType(), request.featureNames(), labelEvents);

            return ResponseEntity.ok(new TrainingDataResponse(
                    rows, rows.size(), request.featureNames().size()));
        } catch (Exception e) {
            return ResponseEntity.internalServerError()
                    .body(new TrainingDataResponse(List.of(), 0, 0));
        }
    }
}
