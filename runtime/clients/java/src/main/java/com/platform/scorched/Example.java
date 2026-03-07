package com.platform.scorched;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Example: connect to scorched, load a dataset, score, and benchmark.
 *
 * <p>Run with:
 * <pre>
 *   mvn compile exec:java -Dexec.mainClass="com.platform.scorched.Example" \
 *       -Dexec.args="localhost 9090"
 * </pre>
 */
public class Example {

    public static void main(String[] args) {
        String host = args.length > 0 ? args[0] : "localhost";
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 9090;
        int numItems = 5000;
        int numItemFeatures = 26;
        int numUserFeatures = 13;
        int topK = 10;
        int benchIterations = 1000;

        try (var client = ScoringClient.connect(host, port)) {
            // Health check
            var health = client.healthz();
            System.out.printf("Health: %s  CUDA: %b  Datasets: %d%n",
                    health.getStatus(), health.getCudaAvailable(), health.getDatasetsLoaded());

            // Build synthetic candidate matrix
            var rng = new Random(42);
            List<Float> matrix = new ArrayList<>(numItems * numItemFeatures);
            for (int i = 0; i < numItems * numItemFeatures; i++) {
                matrix.add(rng.nextFloat());
            }

            // Load dataset
            System.out.printf("Loading %d items (%d features each)...%n", numItems, numItemFeatures);
            var loadResp = client.loadDataset("bench_dataset", matrix, numItems, numItemFeatures, true);
            System.out.printf("Loaded: %s (%d items, %d bytes, %d µs)%n",
                    loadResp.getDatasetName(), loadResp.getNumItems(),
                    loadResp.getMemoryBytes(), loadResp.getLoadDurationUs());

            // Build user vector
            List<Float> userVec = new ArrayList<>(numUserFeatures);
            for (int i = 0; i < numUserFeatures; i++) {
                userVec.add(rng.nextFloat());
            }

            // Score
            var scoreResp = client.scoreTopK("bench_dataset", userVec, topK);
            System.out.printf("%nTop-%d results (%d items scored in %d µs, backend=%s):%n",
                    topK, scoreResp.getItemsScored(), scoreResp.getLatencyUs(), scoreResp.getBackend());
            for (var c : scoreResp.getCandidatesList()) {
                System.out.printf("  [%5d] %.6f%n", c.getIndex(), c.getScore());
            }

            // Benchmark
            System.out.printf("%nBenchmarking %d iterations...%n", benchIterations);
            var result = client.benchmark("bench_dataset", userVec, topK, benchIterations);
            System.out.println(result);

            // List datasets
            var datasets = client.listDatasets();
            System.out.printf("%nLoaded datasets: %d%n", datasets.getDatasetsCount());
            for (var ds : datasets.getDatasetsList()) {
                System.out.printf("  %s: %d items, %d bytes%n",
                        ds.getName(), ds.getNumItems(), ds.getMemoryBytes());
            }
        }
    }
}
