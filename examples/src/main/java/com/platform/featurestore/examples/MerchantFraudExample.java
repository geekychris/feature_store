package com.platform.featurestore.examples;

import com.platform.featurestore.examples.MerchantFraudDataGenerator.*;
import com.platform.featurestore.proto.FeatureVector;
import com.platform.featurestore.proto.GetFeaturesResponse;
import com.platform.featurestore.proto.ViewSchema;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.io.File;
import java.time.OffsetDateTime;
import java.util.*;
import java.util.stream.Collectors;

import static com.platform.featurestore.examples.MerchantFraudDataGenerator.*;

/**
 * End-to-end merchant fraud risk example using the Feature Store Java client.
 * <p>
 * This example demonstrates the complete ML lifecycle:
 * <ol>
 *   <li>Generate 50K synthetic merchant dataset</li>
 *   <li>Register entity, features, and feature view via REST</li>
 *   <li>Materialize feature vectors in bulk (batch write)</li>
 *   <li>Write individual scalar features (demonstrates per-feature updates)</li>
 *   <li>Fetch features from the online store and verify round-trip</li>
 *   <li>Write to Iceberg offline store (attribute form) via REST</li>
 *   <li>Export to Parquet in both attribute and materialized forms</li>
 *   <li>Train XGBoost binary classifier using XGBoost4J</li>
 *   <li>Evaluate model on test set (AUC-ROC, precision, recall)</li>
 *   <li>Run inference: fetch features from store → predict → risk labels</li>
 * </ol>
 *
 * <b>Prerequisites</b>: Feature store server running ({@code docker compose up && mvn spring-boot:run}).
 */
public class MerchantFraudExample {

    private static final int N_MERCHANTS = 50_000;
    private static final long SEED = 42;
    private static final String REST_URL = "http://localhost:8085";
    private static final String GRPC_TARGET = "localhost:9090";

    public static void main(String[] args) throws Exception {
        System.out.println("=".repeat(70));
        System.out.println("ML Feature Store — Java Client Example");
        System.out.println("Merchant Fraud Risk (50K merchants, 15 features, XGBoost)");
        System.out.println("=".repeat(70));

        try (FeatureStoreClient client = new FeatureStoreClient(REST_URL, GRPC_TARGET)) {
            // 1. Generate dataset
            List<MerchantRow> dataset = stepGenerateDataset();

            // 2. Register entity, features, feature view
            String entityId = stepRegister(client);

            // 3. Split into train/test
            List<List<MerchantRow>> splits = trainTestSplit(dataset, 0.2, SEED);
            List<MerchantRow> trainSet = splits.get(0);
            List<MerchantRow> testSet = splits.get(1);
            System.out.printf("  Train: %,d rows (%d positive, %.1f%%)%n",
                    trainSet.size(),
                    trainSet.stream().filter(r -> r.label() == 1).count(),
                    trainSet.stream().filter(r -> r.label() == 1).count() * 100.0 / trainSet.size());
            System.out.printf("  Test:  %,d rows (%d positive)%n",
                    testSet.size(),
                    testSet.stream().filter(r -> r.label() == 1).count());

            // 4. Materialize feature vectors (batch)
            stepMaterializeBatch(client, dataset);

            // 5. Write individual scalar features
            stepScalarWrites(client, dataset);

            // 6. Fetch and verify online features
            stepFetchAndVerify(client, dataset);

            // 7. Write to Iceberg offline store (attribute form)
            stepWriteOffline(client, dataset.subList(0, Math.min(5000, dataset.size())));

            // 8. Export to Parquet
            stepExportParquet(dataset);

            // 9. Train XGBoost model
            Booster model = stepTrainXGBoost(trainSet, testSet);

            // 10. Evaluate on test set
            stepEvaluate(model, testSet);

            // 11. Inference: fetch from store → predict
            stepInference(client, model, testSet);
        }

        System.out.println("\n" + "=".repeat(70));
        System.out.println("Example complete. Verify in admin UI: http://localhost:8086/ui/");
        System.out.println("=".repeat(70));
    }

    // =======================================================================
    // Step 1: Generate Dataset
    // =======================================================================

    private static List<MerchantRow> stepGenerateDataset() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Step 1: Generate Dataset");
        System.out.println("=".repeat(60));

        long t0 = System.currentTimeMillis();
        List<MerchantRow> dataset = MerchantFraudDataGenerator.generate(N_MERCHANTS, SEED);
        long elapsed = System.currentTimeMillis() - t0;

        long positives = dataset.stream().filter(r -> r.label() == 1).count();
        System.out.printf("  Generated %,d merchants in %dms%n", dataset.size(), elapsed);
        System.out.printf("  Positive rate: %d / %d (%.1f%%)%n",
                positives, dataset.size(), positives * 100.0 / dataset.size());
        System.out.printf("  Features: %d (%s ... %s)%n",
                FEATURE_NAMES.size(), FEATURE_NAMES.getFirst(), FEATURE_NAMES.getLast());

        return dataset;
    }

    // =======================================================================
    // Step 2: Register entity, features, feature view
    // =======================================================================

    private static String stepRegister(FeatureStoreClient client) throws Exception {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Step 2: Register Entity, Features, Feature View");
        System.out.println("=".repeat(60));

        // Create or find entity
        String entityId = client.getEntityByName(ENTITY_NAME);
        if (entityId != null) {
            System.out.println("  Entity already exists: " + entityId);
        } else {
            entityId = client.createEntity(ENTITY_NAME,
                    "Merchant entity for fraud risk scoring",
                    "merchant_id", "STRING");
            System.out.println("  Created entity: " + entityId);
        }

        // Create features (skip if they already exist)
        List<Map<String, Object>> existing = client.listFeatures(entityId);
        Set<String> existingNames = existing.stream()
                .map(m -> (String) m.get("name"))
                .collect(Collectors.toSet());

        List<String> featureIds = new ArrayList<>();
        for (FeatureSpec spec : FEATURE_SCHEMA) {
            if (existingNames.contains(spec.name())) {
                String fid = existing.stream()
                        .filter(m -> spec.name().equals(m.get("name")))
                        .map(m -> m.get("id").toString())
                        .findFirst().orElseThrow();
                featureIds.add(fid);
                continue;
            }
            String fid = client.createFeature(spec.name(), entityId, spec.dtype(),
                    spec.description(), "fraud-risk-team", "merchant_risk_daily",
                    spec.updateFrequency(), spec.maxAgeSeconds(), "0.0");
            featureIds.add(fid);
        }
        System.out.printf("  Registered %d features%n", featureIds.size());

        // Create feature view
        try {
            String viewId = client.createFeatureView(VIEW_NAME, VIEW_VERSION, entityId,
                    "Merchant fraud GBDT features (Java example)",
                    "fraud_xgb_java_v1", "XGBOOST", featureIds);
            System.out.println("  Created feature view: " + viewId);
        } catch (Exception e) {
            System.out.println("  Feature view may already exist: " + e.getMessage());
        }

        return entityId;
    }

    // =======================================================================
    // Step 3: Materialize feature vectors (batch)
    // =======================================================================

    private static void stepMaterializeBatch(FeatureStoreClient client,
                                              List<MerchantRow> dataset) throws Exception {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Step 3: Materialize Feature Vectors (batch)");
        System.out.println("=".repeat(60));

        int schemaHash = computeSchemaHash(FEATURE_NAMES);
        long t0 = System.currentTimeMillis();
        int totalMaterialized = 0;

        // Batch in chunks of 500
        int batchSize = 500;
        for (int start = 0; start < dataset.size(); start += batchSize) {
            int end = Math.min(start + batchSize, dataset.size());
            List<Map<String, Object>> batch = new ArrayList<>();

            for (int i = start; i < end; i++) {
                MerchantRow row = dataset.get(i);
                Map<String, Object> vec = new LinkedHashMap<>();
                vec.put("viewName", VIEW_NAME);
                vec.put("viewVersion", VIEW_VERSION);
                vec.put("entityType", ENTITY_NAME);
                vec.put("entityId", row.entityId());
                List<Double> values = new ArrayList<>();
                for (double v : row.features()) values.add(v);
                vec.put("values", values);
                vec.put("schemaHash", schemaHash);
                batch.add(vec);
            }

            client.materializeVectorBatch(batch);
            totalMaterialized += batch.size();

            if (totalMaterialized % 10000 == 0 || totalMaterialized == dataset.size()) {
                System.out.printf("  Materialized %,d / %,d vectors%n",
                        totalMaterialized, dataset.size());
            }
        }

        long elapsed = System.currentTimeMillis() - t0;
        System.out.printf("  Done: %,d vectors in %.1fs (%,.0f vectors/sec)%n",
                totalMaterialized, elapsed / 1000.0, totalMaterialized * 1000.0 / elapsed);
    }

    // =======================================================================
    // Step 4: Write individual scalar features (demonstrates PutScalarFeatures)
    // =======================================================================

    private static void stepScalarWrites(FeatureStoreClient client,
                                          List<MerchantRow> dataset) {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Step 4: Write Individual Scalar Features (gRPC)");
        System.out.println("=".repeat(60));

        // Update a few merchants with individual feature values
        int count = 0;
        for (int i = 0; i < Math.min(10, dataset.size()); i++) {
            MerchantRow row = dataset.get(i);
            Map<String, Double> scalarUpdates = new LinkedHashMap<>();
            for (int f = 0; f < FEATURE_NAMES.size(); f++) {
                scalarUpdates.put(FEATURE_NAMES.get(f), row.features()[f]);
            }
            int written = client.putScalarFeatures(ENTITY_NAME, row.entityId(), scalarUpdates);
            count += written;
        }
        System.out.printf("  Wrote %d scalar features for 10 merchants%n", count);
        System.out.println("  (These are stored separately from vectors in the scalar column family)");
    }

    // =======================================================================
    // Step 5: Fetch features and verify round-trip
    // =======================================================================

    private static void stepFetchAndVerify(FeatureStoreClient client,
                                            List<MerchantRow> dataset) {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Step 5: Fetch Online Features & Verify Round-trip");
        System.out.println("=".repeat(60));

        List<String> sampleIds = dataset.subList(0, 5).stream()
                .map(MerchantRow::entityId).toList();

        GetFeaturesResponse response = client.getOnlineFeatures(VIEW_NAME, VIEW_VERSION, sampleIds);
        System.out.printf("  Fetched %d vectors (latency: %dμs)%n",
                response.getVectorsCount(), response.getLatencyUs());

        // Verify first vector matches input
        if (response.getVectorsCount() > 0) {
            FeatureVector fv = response.getVectors(0);
            MerchantRow expected = dataset.get(0);
            boolean match = true;
            for (int i = 0; i < Math.min(fv.getValuesCount(), expected.features().length); i++) {
                if (Math.abs(fv.getValues(i) - expected.features()[i]) > 0.001) {
                    match = false;
                    break;
                }
            }
            System.out.printf("  Verification for %s: %s%n",
                    fv.getEntityId(), match ? "✓ MATCH" : "✗ MISMATCH");
            System.out.printf("    First 3 values: [%.2f, %.2f, %.2f]%n",
                    fv.getValues(0), fv.getValues(1), fv.getValues(2));
        }

        // Also fetch schema
        try {
            ViewSchema schema = client.getViewSchema(VIEW_NAME, VIEW_VERSION);
            System.out.printf("  Schema: %d features, hash=%d%n",
                    schema.getFeatureNamesCount(), schema.getSchemaHash());
        } catch (Exception e) {
            System.out.println("  Schema fetch: " + e.getMessage());
        }
    }

    // =======================================================================
    // Step 6: Write to Iceberg offline store
    // =======================================================================

    private static void stepWriteOffline(FeatureStoreClient client,
                                          List<MerchantRow> sample) throws Exception {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Step 6: Write to Offline Store (Iceberg attribute form)");
        System.out.println("=".repeat(60));

        String eventTime = OffsetDateTime.now().toString();
        List<Map<String, Object>> records = new ArrayList<>();

        for (MerchantRow row : sample) {
            for (int f = 0; f < FEATURE_NAMES.size(); f++) {
                records.add(Map.of(
                        "entityType", ENTITY_NAME,
                        "entityId", row.entityId(),
                        "featureName", FEATURE_NAMES.get(f),
                        "valueFloat", row.features()[f],
                        "eventTime", eventTime,
                        "pipelineId", "java_example",
                        "viewVersion", VIEW_VERSION
                ));
            }
        }

        long t0 = System.currentTimeMillis();
        int count = client.writeOfflineRecords(records);
        long elapsed = System.currentTimeMillis() - t0;
        System.out.printf("  Wrote %,d feature records (%d merchants × %d features) in %.1fs%n",
                records.size(), sample.size(), FEATURE_NAMES.size(), elapsed / 1000.0);
    }

    // =======================================================================
    // Step 7: Export to Parquet files
    // =======================================================================

    private static void stepExportParquet(List<MerchantRow> dataset) throws Exception {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Step 7: Export to Parquet (attribute + materialized forms)");
        System.out.println("=".repeat(60));

        File exportDir = new File("examples/exports");
        exportDir.mkdirs();

        // Attribute form
        File attrFile = new File(exportDir, "merchant_features_attribute.parquet");
        ParquetExportHelper.writeAttributeForm(dataset, FEATURE_NAMES, attrFile);
        System.out.printf("  Attribute form:      %s (%.1f MB, %,d rows)%n",
                attrFile.getPath(), attrFile.length() / (1024.0 * 1024),
                (long) dataset.size() * FEATURE_NAMES.size());

        // Materialized form
        File matFile = new File(exportDir, "merchant_features_materialized.parquet");
        ParquetExportHelper.writeMaterializedForm(dataset, FEATURE_NAMES, matFile);
        System.out.printf("  Materialized form:   %s (%.1f MB, %,d rows)%n",
                matFile.getPath(), matFile.length() / (1024.0 * 1024), dataset.size());
    }

    // =======================================================================
    // Step 8: Train XGBoost model
    // =======================================================================

    private static Booster stepTrainXGBoost(List<MerchantRow> trainSet,
                                             List<MerchantRow> testSet) throws XGBoostError {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Step 8: Train XGBoost Model (XGBoost4J)");
        System.out.println("=".repeat(60));

        DMatrix trainMatrix = toDMatrix(trainSet);
        DMatrix testMatrix = toDMatrix(testSet);

        Map<String, Object> params = new LinkedHashMap<>();
        params.put("objective", "binary:logistic");
        params.put("eval_metric", "auc");
        params.put("max_depth", 6);
        params.put("learning_rate", 0.1);
        params.put("subsample", 0.8);
        params.put("colsample_bytree", 0.8);
        params.put("min_child_weight", 1);
        params.put("gamma", 0.0);
        params.put("reg_alpha", 0.0);
        params.put("reg_lambda", 1.0);
        params.put("seed", SEED);
        params.put("nthread", Runtime.getRuntime().availableProcessors());

        Map<String, DMatrix> watches = new LinkedHashMap<>();
        watches.put("train", trainMatrix);
        watches.put("test", testMatrix);

        int nRounds = 200;
        long t0 = System.currentTimeMillis();
        Booster booster = XGBoost.train(trainMatrix, params, nRounds, watches, null, null);
        long elapsed = System.currentTimeMillis() - t0;

        System.out.printf("  Trained %d rounds in %.1fs%n", nRounds, elapsed / 1000.0);

        // Save model
        File modelDir = new File("examples/models");
        modelDir.mkdirs();
        File modelFile = new File(modelDir, "fraud_model.ubj");
        booster.saveModel(modelFile.getAbsolutePath());
        System.out.printf("  Model saved: %s (%.0f KB)%n",
                modelFile.getPath(), modelFile.length() / 1024.0);

        return booster;
    }

    // =======================================================================
    // Step 9: Evaluate model
    // =======================================================================

    private static void stepEvaluate(Booster model, List<MerchantRow> testSet) throws XGBoostError {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Step 9: Evaluate Model");
        System.out.println("=".repeat(60));

        DMatrix testMatrix = toDMatrix(testSet);
        float[][] predictions = model.predict(testMatrix);

        // AUC-ROC (simplified — sort-based)
        int n = testSet.size();
        int[] labels = new int[n];
        double[] scores = new double[n];
        for (int i = 0; i < n; i++) {
            labels[i] = testSet.get(i).label();
            scores[i] = predictions[i][0];
        }
        double auc = computeAUC(labels, scores);

        // Confusion matrix at threshold 0.5
        int tp = 0, fp = 0, tn = 0, fn = 0;
        for (int i = 0; i < n; i++) {
            int pred = scores[i] >= 0.5 ? 1 : 0;
            if (pred == 1 && labels[i] == 1) tp++;
            else if (pred == 1 && labels[i] == 0) fp++;
            else if (pred == 0 && labels[i] == 0) tn++;
            else fn++;
        }

        double precision = tp + fp > 0 ? (double) tp / (tp + fp) : 0;
        double recall = tp + fn > 0 ? (double) tp / (tp + fn) : 0;
        double f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;

        System.out.printf("  AUC-ROC:     %.4f%n", auc);
        System.out.printf("  Precision:   %.4f%n", precision);
        System.out.printf("  Recall:      %.4f%n", recall);
        System.out.printf("  F1:          %.4f%n", f1);
        System.out.printf("  Confusion:   TP=%d FP=%d TN=%d FN=%d%n", tp, fp, tn, fn);

        // Risk label distribution
        long high = Arrays.stream(predictions).filter(p -> p[0] >= 0.7).count();
        long medium = Arrays.stream(predictions).filter(p -> p[0] >= 0.3 && p[0] < 0.7).count();
        long low = Arrays.stream(predictions).filter(p -> p[0] < 0.3).count();
        System.out.printf("  Risk labels: HIGH=%d MEDIUM=%d LOW=%d%n", high, medium, low);
    }

    // =======================================================================
    // Step 10: Inference — fetch from store → predict
    // =======================================================================

    private static void stepInference(FeatureStoreClient client, Booster model,
                                       List<MerchantRow> testSet) throws XGBoostError {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Step 10: Inference (fetch from store → predict)");
        System.out.println("=".repeat(60));

        // Pick 20 random test entities
        List<String> sampleIds = testSet.subList(0, Math.min(20, testSet.size())).stream()
                .map(MerchantRow::entityId).toList();

        long t0 = System.currentTimeMillis();
        GetFeaturesResponse response = client.getOnlineFeatures(VIEW_NAME, VIEW_VERSION, sampleIds);
        long fetchMs = System.currentTimeMillis() - t0;

        System.out.printf("  Fetched %d vectors in %dms%n", response.getVectorsCount(), fetchMs);

        if (response.getVectorsCount() == 0) {
            System.out.println("  No vectors returned — skipping inference");
            return;
        }

        // Build DMatrix from fetched vectors
        int nVec = response.getVectorsCount();
        int nFeat = FEATURE_NAMES.size();
        float[] flatData = new float[nVec * nFeat];
        for (int i = 0; i < nVec; i++) {
            FeatureVector fv = response.getVectors(i);
            for (int j = 0; j < Math.min(fv.getValuesCount(), nFeat); j++) {
                flatData[i * nFeat + j] = (float) fv.getValues(j);
            }
        }

        DMatrix scoringMatrix = new DMatrix(flatData, nVec, nFeat);
        float[][] preds = model.predict(scoringMatrix);
        long scoreMs = System.currentTimeMillis() - t0 - fetchMs;

        System.out.printf("  Scored %d entities in %dms%n", nVec, scoreMs);
        System.out.println("  Results:");
        for (int i = 0; i < Math.min(10, nVec); i++) {
            FeatureVector fv = response.getVectors(i);
            float prob = preds[i][0];
            String risk = prob >= 0.7 ? "HIGH" : prob >= 0.3 ? "MEDIUM" : "LOW";
            System.out.printf("    %s: risk=%-6s (prob=%.4f) gmv_30d=%.0f chargeback=%.4f%n",
                    fv.getEntityId(), risk, prob, fv.getValues(0), fv.getValues(5));
        }
    }

    // =======================================================================
    // Helpers
    // =======================================================================

    private static DMatrix toDMatrix(List<MerchantRow> rows) throws XGBoostError {
        int n = rows.size();
        int nFeat = FEATURE_NAMES.size();
        float[] data = new float[n * nFeat];
        float[] labels = new float[n];

        for (int i = 0; i < n; i++) {
            MerchantRow row = rows.get(i);
            for (int j = 0; j < nFeat; j++) {
                data[i * nFeat + j] = (float) row.features()[j];
            }
            labels[i] = row.label();
        }

        DMatrix matrix = new DMatrix(data, n, nFeat);
        matrix.setLabel(labels);
        return matrix;
    }

    /** Compute AUC-ROC via the sorted-pairs method. */
    private static double computeAUC(int[] labels, double[] scores) {
        int n = labels.length;
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Double.compare(scores[b], scores[a]));

        long posTotal = Arrays.stream(labels).filter(l -> l == 1).count();
        long negTotal = n - posTotal;
        if (posTotal == 0 || negTotal == 0) return 0.5;

        long tp = 0, fp = 0;
        double auc = 0;
        double prevFpr = 0, prevTpr = 0;

        for (int idx : indices) {
            if (labels[idx] == 1) {
                tp++;
            } else {
                fp++;
            }
            double tpr = (double) tp / posTotal;
            double fpr = (double) fp / negTotal;
            auc += (fpr - prevFpr) * (tpr + prevTpr) / 2.0;
            prevFpr = fpr;
            prevTpr = tpr;
        }
        return auc;
    }
}
