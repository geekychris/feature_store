package com.platform.featurestore.service;

import com.platform.featurestore.proto.FeatureVector;
import com.platform.featurestore.proto.ViewSchema;
import com.platform.featurestore.store.online.RocksDBFeatureStore;
import org.rocksdb.RocksDBException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.*;

/**
 * Validation service: schema consistency, staleness SLAs, feature drift.
 */
@Service
public class ValidationService {

    private static final Logger log = LoggerFactory.getLogger(ValidationService.class);

    // Validation gate thresholds
    private static final double AUC_ROC_THRESHOLD = 0.75;
    private static final double AUC_PR_THRESHOLD = 0.40;
    private static final double SCORE_STD_THRESHOLD = 0.05;
    private static final double DEGRADATION_THRESHOLD = 0.02;

    private final RocksDBFeatureStore rocksStore;

    public ValidationService(RocksDBFeatureStore rocksStore) {
        this.rocksStore = rocksStore;
    }

    /**
     * Validate that a feature vector's schema hash matches the expected schema.
     */
    public ValidationResult validateSchemaHash(FeatureVector vector, ViewSchema expectedSchema) {
        List<ValidationGate> gates = new ArrayList<>();

        boolean hashMatch = vector.getSchemaHash() == expectedSchema.getSchemaHash();
        gates.add(new ValidationGate(
                "Schema Hash Match",
                hashMatch,
                hashMatch ? 1.0 : 0.0,
                1.0,
                hashMatch ? "Schema hash matches"
                        : "Schema mismatch: vector=" + vector.getSchemaHash()
                          + " expected=" + expectedSchema.getSchemaHash()
        ));

        // Vector length check
        boolean lengthMatch = vector.getValuesCount() == expectedSchema.getFeatureNamesCount();
        gates.add(new ValidationGate(
                "Vector Length",
                lengthMatch,
                vector.getValuesCount(),
                expectedSchema.getFeatureNamesCount(),
                lengthMatch ? "Vector length matches schema"
                        : "Length mismatch: got " + vector.getValuesCount()
                          + " expected " + expectedSchema.getFeatureNamesCount()
        ));

        return new ValidationResult("SCHEMA_VALIDATION", gates, List.of());
    }

    /**
     * Validate staleness of feature values against SLA.
     */
    public ValidationResult validateStaleness(FeatureVector vector,
                                               Map<String, Integer> maxAgeSecondsMap,
                                               List<String> featureNames) {
        List<ValidationGate> gates = new ArrayList<>();
        List<String> warnings = new ArrayList<>();
        long nowMs = System.currentTimeMillis();

        for (int i = 0; i < featureNames.size() && i < vector.getValueAgesMsCount(); i++) {
            String fname = featureNames.get(i);
            long ageMs = vector.getValueAgesMs(i);

            if (ageMs < 0) {
                warnings.add("Feature '" + fname + "' has unknown age (default value?)");
                continue;
            }

            Integer maxAgeSeconds = maxAgeSecondsMap.get(fname);
            if (maxAgeSeconds != null) {
                long maxAgeMs = maxAgeSeconds * 1000L;
                boolean fresh = ageMs <= maxAgeMs;
                gates.add(new ValidationGate(
                        "Staleness: " + fname,
                        fresh,
                        ageMs / 1000.0,
                        maxAgeSeconds,
                        fresh ? "Within SLA" : "STALE: " + (ageMs / 1000) + "s > " + maxAgeSeconds + "s"
                ));
            }
        }

        // Check default count
        long defaultCount = vector.getIsDefaultMaskList().stream().filter(b -> b).count();
        if (defaultCount > 0) {
            double defaultRate = (double) defaultCount / vector.getValuesCount();
            warnings.add(String.format("%.1f%% of features are default-filled (%d/%d)",
                    defaultRate * 100, defaultCount, vector.getValuesCount()));
        }

        return new ValidationResult("STALENESS_VALIDATION", gates, warnings);
    }

    /**
     * Validate model performance metrics against gates.
     */
    public ValidationResult validateModelMetrics(double aucRoc, double aucPr,
                                                   double scoreStd,
                                                   Double baselineAucRoc) {
        List<ValidationGate> gates = new ArrayList<>();
        List<String> warnings = new ArrayList<>();

        gates.add(new ValidationGate(
                "AUC-ROC", aucRoc >= AUC_ROC_THRESHOLD,
                aucRoc, AUC_ROC_THRESHOLD,
                "Area under ROC curve"
        ));

        gates.add(new ValidationGate(
                "AUC-PR", aucPr >= AUC_PR_THRESHOLD,
                aucPr, AUC_PR_THRESHOLD,
                "Area under Precision-Recall curve"
        ));

        gates.add(new ValidationGate(
                "Score distribution std", scoreStd > SCORE_STD_THRESHOLD,
                scoreStd, SCORE_STD_THRESHOLD,
                "Model produces differentiated scores"
        ));

        if (baselineAucRoc != null) {
            double degradation = baselineAucRoc - aucRoc;
            gates.add(new ValidationGate(
                    "vs Baseline regression",
                    degradation < DEGRADATION_THRESHOLD,
                    degradation, DEGRADATION_THRESHOLD,
                    "AUC degradation vs baseline (baseline: " + String.format("%.4f", baselineAucRoc) + ")"
            ));
        }

        return new ValidationResult("MODEL_VALIDATION", gates, warnings);
    }

    /**
     * Validate a feature vector retrieved from the store.
     */
    public ValidationResult validateFeatureVector(String viewName, int viewVersion,
                                                    String entityId) {
        try {
            Optional<ViewSchema> schema = rocksStore.getSchema(viewName, viewVersion);
            if (schema.isEmpty()) {
                return new ValidationResult("VALIDATION_ERROR",
                        List.of(new ValidationGate("Schema exists", false, 0, 1,
                                "No schema found for " + viewName + ":" + viewVersion)),
                        List.of());
            }

            Optional<FeatureVector> vector = rocksStore.getFeatureVector(viewName, viewVersion, entityId);
            if (vector.isEmpty()) {
                return new ValidationResult("VALIDATION_ERROR",
                        List.of(new ValidationGate("Vector exists", false, 0, 1,
                                "No vector found for " + entityId)),
                        List.of());
            }

            return validateSchemaHash(vector.get(), schema.get());
        } catch (RocksDBException e) {
            return new ValidationResult("VALIDATION_ERROR",
                    List.of(new ValidationGate("Store access", false, 0, 1,
                            "RocksDB error: " + e.getMessage())),
                    List.of());
        }
    }

    // Result types

    public record ValidationGate(
            String name,
            boolean passed,
            double metric,
            double threshold,
            String message
    ) {}

    public record ValidationResult(
            String type,
            List<ValidationGate> gates,
            List<String> warnings
    ) {
        public boolean passed() {
            return gates.stream().allMatch(ValidationGate::passed);
        }
    }
}
