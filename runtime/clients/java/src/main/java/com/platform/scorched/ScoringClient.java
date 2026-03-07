package com.platform.scorched;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * High-level client for the scorched scoring runtime.
 *
 * <p>Usage:
 * <pre>
 *   try (var client = ScoringClient.connect("localhost", 9090)) {
 *       var resp = client.scoreTopK("ads_us", userVector, 10);
 *       resp.getCandidatesList().forEach(c ->
 *           System.out.printf("[%d] %.6f%n", c.getIndex(), c.getScore()));
 *   }
 * </pre>
 */
public class ScoringClient implements AutoCloseable {

    private final ManagedChannel channel;
    private final ScoreServiceGrpc.ScoreServiceBlockingStub scoreStub;
    private final DatasetServiceGrpc.DatasetServiceBlockingStub datasetStub;
    private final StatsServiceGrpc.StatsServiceBlockingStub statsStub;

    private ScoringClient(ManagedChannel channel) {
        this.channel = channel;
        this.scoreStub = ScoreServiceGrpc.newBlockingStub(channel);
        this.datasetStub = DatasetServiceGrpc.newBlockingStub(channel);
        this.statsStub = StatsServiceGrpc.newBlockingStub(channel);
    }

    /** Connect to scorched at the given host and port. */
    public static ScoringClient connect(String host, int port) {
        ManagedChannel ch = ManagedChannelBuilder.forAddress(host, port)
                .usePlaintext()
                .maxInboundMessageSize(64 * 1024 * 1024)
                .build();
        return new ScoringClient(ch);
    }

    // ---- ScoreService ----

    /** Score a user vector against a named dataset, returning top-K candidates. */
    public ScorchedProto.ScoreResponse scoreTopK(String dataset, List<Float> userVector, int topK) {
        return scoreStub.scoreTopK(ScorchedProto.ScoreRequest.newBuilder()
                .setDatasetName(dataset)
                .addAllUserVector(userVector)
                .setTopK(topK)
                .build());
    }

    /** Score a user vector with a custom request ID. */
    public ScorchedProto.ScoreResponse scoreTopK(String dataset, List<Float> userVector, int topK, String requestId) {
        return scoreStub.scoreTopK(ScorchedProto.ScoreRequest.newBuilder()
                .setDatasetName(dataset)
                .addAllUserVector(userVector)
                .setTopK(topK)
                .setRequestId(requestId)
                .build());
    }

    /** Batch-score multiple users against a dataset. */
    public ScorchedProto.ScoreBatchResponse scoreBatch(String dataset,
                                                        List<ScorchedProto.UserScoreRequest> users,
                                                        int topK) {
        return scoreStub.scoreBatch(ScorchedProto.ScoreBatchRequest.newBuilder()
                .setDatasetName(dataset)
                .addAllUsers(users)
                .setTopK(topK)
                .build());
    }

    // ---- DatasetService ----

    /** Load a CandidateMatrix dataset. */
    public ScorchedProto.LoadDatasetResponse loadDataset(String name,
                                                          List<Float> candidateMatrix,
                                                          int numItems,
                                                          int numItemFeatures,
                                                          boolean replaceIfExists) {
        return datasetStub.loadDataset(ScorchedProto.LoadDatasetRequest.newBuilder()
                .setName(name)
                .addAllCandidateMatrix(candidateMatrix)
                .setNumItems(numItems)
                .setNumItemFeatures(numItemFeatures)
                .setReplaceIfExists(replaceIfExists)
                .build());
    }

    /** Load a dataset from a server-local file path. */
    public ScorchedProto.LoadDatasetResponse loadFromFile(String name, String filePath,
                                                           int numItemFeatures, boolean replaceIfExists) {
        return datasetStub.loadDatasetFromFile(ScorchedProto.LoadDatasetFromFileRequest.newBuilder()
                .setName(name)
                .setFilePath(filePath)
                .setNumItemFeatures(numItemFeatures)
                .setReplaceIfExists(replaceIfExists)
                .build());
    }

    /** Unload a named dataset. */
    public ScorchedProto.UnloadDatasetResponse unloadDataset(String name) {
        return datasetStub.unloadDataset(ScorchedProto.UnloadDatasetRequest.newBuilder()
                .setName(name)
                .build());
    }

    /** List all loaded datasets. */
    public ScorchedProto.ListDatasetsResponse listDatasets() {
        return datasetStub.listDatasets(ScorchedProto.ListDatasetsRequest.getDefaultInstance());
    }

    /** Get info about a specific dataset. */
    public ScorchedProto.DatasetInfoResponse getDatasetInfo(String name) {
        return datasetStub.getDatasetInfo(ScorchedProto.DatasetInfoRequest.newBuilder()
                .setName(name)
                .build());
    }

    // ---- StatsService ----

    /** Get runtime statistics. */
    public ScorchedProto.StatsResponse getStats() {
        return statsStub.getStats(ScorchedProto.GetStatsRequest.getDefaultInstance());
    }

    /** Health check. */
    public ScorchedProto.HealthResponse healthz() {
        return statsStub.getHealthz(ScorchedProto.GetHealthzRequest.getDefaultInstance());
    }

    // ---- Benchmark ----

    /** Run a simple latency benchmark. */
    public BenchmarkResult benchmark(String dataset, List<Float> userVector, int topK, int iterations) {
        // Warmup
        for (int i = 0; i < 10; i++) {
            scoreTopK(dataset, userVector, topK);
        }

        long[] latencies = new long[iterations];
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            long t = System.nanoTime();
            scoreTopK(dataset, userVector, topK);
            latencies[i] = System.nanoTime() - t;
        }
        long totalNs = System.nanoTime() - start;
        java.util.Arrays.sort(latencies);

        return new BenchmarkResult(
                iterations,
                totalNs,
                latencies[iterations / 2],
                latencies[(int) (iterations * 0.95)],
                latencies[(int) (iterations * 0.99)],
                iterations / (totalNs / 1_000_000_000.0)
        );
    }

    @Override
    public void close() {
        try {
            channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            channel.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    /** Benchmark results with percentile latencies. */
    public record BenchmarkResult(
            int requests,
            long totalNanos,
            long p50Nanos,
            long p95Nanos,
            long p99Nanos,
            double qps
    ) {
        @Override
        public String toString() {
            return String.format(
                    "BenchmarkResult{requests=%d, total=%.1fms, p50=%.3fms, p95=%.3fms, p99=%.3fms, qps=%.1f}",
                    requests,
                    totalNanos / 1_000_000.0,
                    p50Nanos / 1_000_000.0,
                    p95Nanos / 1_000_000.0,
                    p99Nanos / 1_000_000.0,
                    qps
            );
        }
    }
}
