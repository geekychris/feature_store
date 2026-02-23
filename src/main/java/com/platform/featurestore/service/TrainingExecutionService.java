package com.platform.featurestore.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.platform.featurestore.service.DatasetDiscoveryService.DatasetDescriptor;
import com.platform.featurestore.service.DatasetDiscoveryService.ParamDescriptor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

/**
 * Manages execution of Python training and dataset import as subprocesses.
 * Dataset definitions are provided by {@link DatasetDiscoveryService} — no
 * hardcoded dataset knowledge lives here.
 */
@Service
public class TrainingExecutionService {

    private static final Logger log = LoggerFactory.getLogger(TrainingExecutionService.class);
    private static final ObjectMapper mapper = new ObjectMapper();

    private final AtomicReference<RunStatus> currentStatus = new AtomicReference<>(RunStatus.IDLE);
    private final List<TrainingRun> runHistory = new CopyOnWriteArrayList<>();
    private volatile Process currentProcess;

    public enum RunStatus {
        IDLE, RUNNING, COMPLETED, FAILED
    }

    /**
     * Training configuration driven entirely by the dataset descriptor.
     *
     * @param dataset        descriptor loaded from dataset_manifest.json
     * @param standalone     run without feature store backend
     * @param noSearch       skip hyperparameter search
     * @param datasetParams  per-dataset params (keys match ParamDescriptor.name)
     * @param hyperparams    XGBoost hyperparams (max_depth, learning_rate, …)
     */
    public record TrainingConfig(
            DatasetDescriptor dataset,
            boolean standalone,
            boolean noSearch,
            Map<String, Object> datasetParams,
            Map<String, Object> hyperparams
    ) {
        public Map<String, Object> toDisplayMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("dataset", dataset.name());
            map.put("standalone", standalone);
            map.put("no_search", noSearch);
            map.putAll(datasetParams);
            if (!hyperparams.isEmpty()) {
                map.putAll(hyperparams);
            }
            return map;
        }
    }

    public record TrainingRun(
            String id,
            Instant startedAt,
            Instant completedAt,
            RunStatus status,
            Map<String, Object> config,
            Map<String, Object> results,
            List<String> logLines
    ) {}

    public RunStatus getStatus() {
        return currentStatus.get();
    }

    public List<TrainingRun> getRunHistory() {
        return Collections.unmodifiableList(runHistory);
    }

    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    public String startTraining(TrainingConfig config, Consumer<String> logConsumer) {
        if (currentStatus.get() == RunStatus.RUNNING) {
            throw new IllegalStateException("A training run is already in progress");
        }

        String runId = UUID.randomUUID().toString().substring(0, 8);
        currentStatus.set(RunStatus.RUNNING);
        List<String> logLines = new CopyOnWriteArrayList<>();

        Thread.ofVirtual().name("training-" + runId).start(() -> {
            Instant startedAt = Instant.now();
            RunStatus finalStatus = RunStatus.FAILED;
            Map<String, Object> results = new HashMap<>();

            try {
                String projectDir = findProjectDir();
                List<String> command = buildTrainingCommand(config);

                ProcessBuilder pb = new ProcessBuilder(command);
                pb.directory(new File(projectDir));
                pb.redirectErrorStream(true);

                emitLog(logConsumer, logLines, ">>> Training run " + runId + " — " + config.dataset().name());
                emitLog(logConsumer, logLines, ">>> Command: " + String.join(" ", command));
                emitLog(logConsumer, logLines, ">>> Working directory: " + projectDir);
                emitLog(logConsumer, logLines, "");

                currentProcess = pb.start();

                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(currentProcess.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        emitLog(logConsumer, logLines, line);
                    }
                }

                int exitCode = currentProcess.waitFor();
                emitLog(logConsumer, logLines, "");
                emitLog(logConsumer, logLines, ">>> Process exited with code: " + exitCode);

                if (exitCode == 0) {
                    finalStatus = RunStatus.COMPLETED;
                    results = tryParseResults(projectDir, config.dataset());
                } else {
                    finalStatus = RunStatus.FAILED;
                }

            } catch (Exception e) {
                log.error("Training run {} failed", runId, e);
                emitLog(logConsumer, logLines, ">>> ERROR: " + e.getMessage());
            } finally {
                currentProcess = null;
                currentStatus.set(finalStatus);

                TrainingRun run = new TrainingRun(
                        runId, startedAt, Instant.now(), finalStatus,
                        config.toDisplayMap(), results, logLines
                );
                runHistory.addFirst(run);
                while (runHistory.size() > 20) runHistory.removeLast();

                emitLog(logConsumer, logLines, ">>> Run " + runId + " finished: " + finalStatus);
            }
        });

        return runId;
    }

    // -----------------------------------------------------------------------
    // Dataset Import
    // -----------------------------------------------------------------------

    public String startDatasetImport(DatasetDescriptor dataset, Map<String, Object> params,
                                      Consumer<String> logConsumer) {
        if (currentStatus.get() == RunStatus.RUNNING) {
            throw new IllegalStateException("A process is already in progress");
        }

        String runId = "imp-" + UUID.randomUUID().toString().substring(0, 6);
        currentStatus.set(RunStatus.RUNNING);
        List<String> logLines = new CopyOnWriteArrayList<>();

        Thread.ofVirtual().name("import-" + runId).start(() -> {
            Instant startedAt = Instant.now();
            RunStatus finalStatus = RunStatus.FAILED;

            try {
                String projectDir = findProjectDir();
                List<String> command = buildImportCommand(dataset, params);

                ProcessBuilder pb = new ProcessBuilder(command);
                pb.directory(new File(projectDir));
                pb.redirectErrorStream(true);

                emitLog(logConsumer, logLines, ">>> Import " + runId + " — " + dataset.name());
                emitLog(logConsumer, logLines, ">>> Command: " + String.join(" ", command));
                emitLog(logConsumer, logLines, "");

                currentProcess = pb.start();

                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(currentProcess.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        emitLog(logConsumer, logLines, line);
                    }
                }

                int exitCode = currentProcess.waitFor();
                emitLog(logConsumer, logLines, "");
                emitLog(logConsumer, logLines, ">>> Process exited with code: " + exitCode);
                finalStatus = exitCode == 0 ? RunStatus.COMPLETED : RunStatus.FAILED;

            } catch (Exception e) {
                log.error("Import {} failed", runId, e);
                emitLog(logConsumer, logLines, ">>> ERROR: " + e.getMessage());
            } finally {
                currentProcess = null;
                currentStatus.set(finalStatus);

                Map<String, Object> config = new LinkedHashMap<>();
                config.put("type", "import");
                config.put("dataset", dataset.name());
                config.putAll(params);

                TrainingRun run = new TrainingRun(
                        runId, startedAt, Instant.now(), finalStatus,
                        config, Map.of(), logLines
                );
                runHistory.addFirst(run);
                while (runHistory.size() > 20) runHistory.removeLast();

                emitLog(logConsumer, logLines, ">>> Import " + runId + " finished: " + finalStatus);
            }
        });

        return runId;
    }

    // -----------------------------------------------------------------------
    // Dataset Preview
    // -----------------------------------------------------------------------

    /**
     * Run dataset_tool.py preview and capture the __PREVIEW_JSON__ output.
     */
    public void startDatasetPreview(DatasetDescriptor dataset, Map<String, Object> params,
                                     int rows, Consumer<String> logConsumer,
                                     Consumer<String> jsonConsumer) {
        if (currentStatus.get() == RunStatus.RUNNING) {
            throw new IllegalStateException("A process is already in progress");
        }

        String runId = "prev-" + UUID.randomUUID().toString().substring(0, 6);
        currentStatus.set(RunStatus.RUNNING);
        List<String> logLines = new CopyOnWriteArrayList<>();

        Thread.ofVirtual().name("preview-" + runId).start(() -> {
            try {
                String projectDir = findProjectDir();
                List<String> command = buildDatasetToolCommand("preview", dataset, params);
                command.add("--rows");
                command.add(String.valueOf(rows));

                ProcessBuilder pb = new ProcessBuilder(command);
                pb.directory(new File(projectDir));
                pb.redirectErrorStream(true);

                emitLog(logConsumer, logLines, ">>> Preview " + runId + " — " + dataset.name());
                emitLog(logConsumer, logLines, ">>> Command: " + String.join(" ", command));

                currentProcess = pb.start();

                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(currentProcess.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        if (line.startsWith("__PREVIEW_JSON__:")) {
                            jsonConsumer.accept(line.substring("__PREVIEW_JSON__:".length()));
                        } else {
                            emitLog(logConsumer, logLines, line);
                        }
                    }
                }

                int exitCode = currentProcess.waitFor();
                emitLog(logConsumer, logLines, ">>> Preview exited with code: " + exitCode);

            } catch (Exception e) {
                log.error("Preview {} failed", runId, e);
                emitLog(logConsumer, logLines, ">>> ERROR: " + e.getMessage());
            } finally {
                currentProcess = null;
                currentStatus.set(RunStatus.IDLE);
            }
        });
    }

    // -----------------------------------------------------------------------
    // Dataset Export
    // -----------------------------------------------------------------------

    /**
     * Run dataset_tool.py export (parquet or iceberg).
     */
    public void startDatasetExport(DatasetDescriptor dataset, Map<String, Object> params,
                                    String format, Consumer<String> logConsumer) {
        if (currentStatus.get() == RunStatus.RUNNING) {
            throw new IllegalStateException("A process is already in progress");
        }

        String runId = "exp-" + UUID.randomUUID().toString().substring(0, 6);
        currentStatus.set(RunStatus.RUNNING);
        List<String> logLines = new CopyOnWriteArrayList<>();

        Thread.ofVirtual().name("export-" + runId).start(() -> {
            RunStatus finalStatus = RunStatus.FAILED;
            try {
                String projectDir = findProjectDir();
                List<String> command = buildDatasetToolCommand("export", dataset, params);
                command.add("--format");
                command.add(format);

                ProcessBuilder pb = new ProcessBuilder(command);
                pb.directory(new File(projectDir));
                pb.redirectErrorStream(true);

                emitLog(logConsumer, logLines, ">>> Export " + runId + " — " + dataset.name() + " (" + format + ")");
                emitLog(logConsumer, logLines, ">>> Command: " + String.join(" ", command));

                currentProcess = pb.start();

                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(currentProcess.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        emitLog(logConsumer, logLines, line);
                    }
                }

                int exitCode = currentProcess.waitFor();
                emitLog(logConsumer, logLines, ">>> Export exited with code: " + exitCode);
                finalStatus = exitCode == 0 ? RunStatus.COMPLETED : RunStatus.FAILED;

            } catch (Exception e) {
                log.error("Export {} failed", runId, e);
                emitLog(logConsumer, logLines, ">>> ERROR: " + e.getMessage());
            } finally {
                currentProcess = null;
                currentStatus.set(finalStatus);
                emitLog(logConsumer, logLines, ">>> Export " + runId + " finished: " + currentStatus.get());
            }
        });
    }

    // -----------------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------------

    /**
     * Run dataset_tool.py infer and capture both __SAMPLE_JSON__ and __INFER_JSON__ output.
     *
     * @param inputFile  if non-null, pass --input-file to score edited data from this file
     */
    public void startInference(DatasetDescriptor dataset, int nSamples,
                                Path inputFile,
                                Consumer<String> logConsumer,
                                Consumer<String> sampleConsumer,
                                Consumer<String> jsonConsumer) {
        if (currentStatus.get() == RunStatus.RUNNING) {
            throw new IllegalStateException("A process is already in progress");
        }

        String runId = "inf-" + UUID.randomUUID().toString().substring(0, 6);
        currentStatus.set(RunStatus.RUNNING);
        List<String> logLines = new CopyOnWriteArrayList<>();

        Thread.ofVirtual().name("infer-" + runId).start(() -> {
            Instant startedAt = Instant.now();
            RunStatus finalStatus = RunStatus.FAILED;

            try {
                String projectDir = findProjectDir();
                // Build command directly to avoid duplicate --n-samples from dataset params
                List<String> command = new ArrayList<>();
                command.add("python3");
                command.add("python/dataset_tool.py");
                command.add("infer");
                command.add("--dataset");
                command.add(dataset.id());
                if (inputFile != null) {
                    command.add("--input-file");
                    command.add(inputFile.toAbsolutePath().toString());
                } else {
                    command.add("--n-samples");
                    command.add(String.valueOf(nSamples));
                }

                ProcessBuilder pb = new ProcessBuilder(command);
                pb.directory(new File(projectDir));
                pb.redirectErrorStream(true);

                emitLog(logConsumer, logLines, ">>> Inference " + runId + " — " + dataset.name());
                emitLog(logConsumer, logLines, ">>> Command: " + String.join(" ", command));

                currentProcess = pb.start();

                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(currentProcess.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        if (line.startsWith("__SAMPLE_JSON__:")) {
                            sampleConsumer.accept(line.substring("__SAMPLE_JSON__:".length()));
                        } else if (line.startsWith("__INFER_JSON__:")) {
                            jsonConsumer.accept(line.substring("__INFER_JSON__:".length()));
                        } else {
                            emitLog(logConsumer, logLines, line);
                        }
                    }
                }

                int exitCode = currentProcess.waitFor();
                emitLog(logConsumer, logLines, ">>> Inference exited with code: " + exitCode);
                finalStatus = exitCode == 0 ? RunStatus.COMPLETED : RunStatus.FAILED;

            } catch (Exception e) {
                log.error("Inference {} failed", runId, e);
                emitLog(logConsumer, logLines, ">>> ERROR: " + e.getMessage());
            } finally {
                currentProcess = null;
                currentStatus.set(finalStatus);

                Map<String, Object> config = new LinkedHashMap<>();
                config.put("type", "inference");
                config.put("dataset", dataset.name());

                TrainingRun run = new TrainingRun(
                        runId, startedAt, Instant.now(), finalStatus,
                        config, Map.of(), logLines
                );
                runHistory.addFirst(run);
                while (runHistory.size() > 20) runHistory.removeLast();

                emitLog(logConsumer, logLines, ">>> Inference " + runId + " finished: " + finalStatus);
            }
        });
    }

    // -----------------------------------------------------------------------
    // Cancel
    // -----------------------------------------------------------------------

    public void cancelTraining() {
        Process p = currentProcess;
        if (p != null && p.isAlive()) {
            p.destroyForcibly();
            currentStatus.set(RunStatus.FAILED);
        }
    }

    // -----------------------------------------------------------------------
    // Command builders (fully driven by descriptors — no switch/case)
    // -----------------------------------------------------------------------

    private List<String> buildTrainingCommand(TrainingConfig config) {
        DatasetDescriptor ds = config.dataset();
        List<String> cmd = new ArrayList<>();
        cmd.add("python3");
        cmd.add(ds.trainScript());

        if (config.standalone()) cmd.add("--standalone");
        if (config.noSearch()) cmd.add("--no-search");

        // Ensure model is saved to the manifest-specified models_dir
        if (ds.modelsDir() != null && !ds.modelsDir().isBlank()) {
            cmd.add("--model-dir");
            cmd.add(ds.modelsDir());
        }

        // Append per-dataset params using CLI args from the manifest
        for (ParamDescriptor param : ds.params()) {
            Object value = config.datasetParams().get(param.name());
            if (value != null && param.cliArg() != null) {
                cmd.add(param.cliArg());
                cmd.add(value.toString());
            }
        }

        return cmd;
    }

    private List<String> buildImportCommand(DatasetDescriptor dataset, Map<String, Object> params) {
        List<String> cmd = new ArrayList<>();
        cmd.add("python3");
        cmd.add("python/import_dataset.py");
        cmd.add("--dataset");
        cmd.add(dataset.importId());

        // Append per-dataset params using CLI args from the manifest
        for (ParamDescriptor param : dataset.params()) {
            Object value = params.get(param.name());
            if (value != null && param.cliArg() != null) {
                cmd.add(param.cliArg());
                cmd.add(value.toString());
            }
        }

        Object maxVec = params.get("max_vectors");
        if (maxVec != null) { cmd.add("--max-vectors"); cmd.add(maxVec.toString()); }

        return cmd;
    }

    // -----------------------------------------------------------------------
    // dataset_tool.py command builder
    // -----------------------------------------------------------------------

    private List<String> buildDatasetToolCommand(String subcommand, DatasetDescriptor dataset,
                                                  Map<String, Object> params) {
        List<String> cmd = new ArrayList<>();
        cmd.add("python3");
        cmd.add("python/dataset_tool.py");
        cmd.add(subcommand);
        cmd.add("--dataset");
        cmd.add(dataset.id());

        // Append per-dataset params using CLI args from the manifest
        for (ParamDescriptor param : dataset.params()) {
            Object value = params.get(param.name());
            if (value != null && param.cliArg() != null) {
                cmd.add(param.cliArg());
                cmd.add(value.toString());
            }
        }
        return cmd;
    }

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    private void emitLog(Consumer<String> logConsumer, List<String> logLines, String line) {
        logLines.add(line);
        try {
            logConsumer.accept(line);
        } catch (Exception e) {
            log.debug("Log consumer error: {}", e.getMessage());
        }
    }

    private String findProjectDir() {
        String cwd = System.getProperty("user.dir");
        Path path = Path.of(cwd);

        if (Files.isDirectory(path.resolve("python"))) {
            return path.toString();
        }

        Path home = Path.of(System.getProperty("user.home"));
        Path candidate = home.resolve("code/warp_experiments/feature_store");
        if (Files.isDirectory(candidate.resolve("python"))) {
            return candidate.toString();
        }

        return cwd;
    }

    /**
     * Parse training results generically — reads all numeric fields from training_result.json
     * plus special fields like gates_passed and feature_importance.
     */
    private Map<String, Object> tryParseResults(String projectDir, DatasetDescriptor dataset) {
        try {
            Path resultsPath = Path.of(projectDir, dataset.modelsDir());
            if (!Files.isDirectory(resultsPath)) {
                return Map.of();
            }

            Optional<Path> resultFile = Files.walk(resultsPath, 2)
                    .filter(p -> p.getFileName().toString().equals("training_result.json"))
                    .max(Comparator.comparingLong(p -> {
                        try { return Files.getLastModifiedTime(p).toMillis(); }
                        catch (Exception e) { return 0L; }
                    }));

            if (resultFile.isPresent()) {
                JsonNode node = mapper.readTree(resultFile.get().toFile());
                Map<String, Object> results = new HashMap<>();

                // Extract all numeric fields generically
                node.fields().forEachRemaining(entry -> {
                    JsonNode val = entry.getValue();
                    if (val.isNumber()) {
                        results.put(entry.getKey(), val.asDouble());
                    } else if (val.isBoolean()) {
                        results.put(entry.getKey(), val.asBoolean());
                    }
                });

                // Feature importance (object/map)
                if (node.has("feature_importance")) {
                    results.put("feature_importance", mapper.convertValue(
                            node.get("feature_importance"), Map.class));
                }
                return results;
            }
        } catch (Exception e) {
            log.warn("Could not parse training results: {}", e.getMessage());
        }
        return Map.of();
    }
}
