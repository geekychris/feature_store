package com.platform.featurestore.service;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Stream;

/**
 * Discovers available ML datasets by scanning python/*_example/dataset_manifest.json files.
 *
 * Convention: any directory matching python/*_example/ that contains a dataset_manifest.json
 * is treated as a dataset. The manifest fully describes the dataset's metadata, CLI params,
 * training script, and expected metrics.
 */
@Service
public class DatasetDiscoveryService {

    private static final Logger log = LoggerFactory.getLogger(DatasetDiscoveryService.class);
    private static final ObjectMapper mapper = new ObjectMapper();
    private static final String MANIFEST_FILENAME = "dataset_manifest.json";

    private final List<DatasetDescriptor> datasets = new CopyOnWriteArrayList<>();

    // -----------------------------------------------------------------------
    // Data model
    // -----------------------------------------------------------------------

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record DatasetDescriptor(
            String id,
            String name,
            String description,
            @JsonProperty("train_script") String trainScript,
            @JsonProperty("import_id") String importId,
            @JsonProperty("models_dir") String modelsDir,
            @JsonProperty("infer_script") String inferScript,
            @JsonProperty("infer_type") String inferType,
            List<ParamDescriptor> params,
            List<String> metrics,
            @JsonProperty("primary_metric") String primaryMetric
    ) {
        @Override
        public String toString() {
            return name;
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ParamDescriptor(
            String name,
            String label,
            String type,
            @JsonProperty("default") Number defaultValue,
            Number min,
            Number max,
            Number step,
            @JsonProperty("cli_arg") String cliArg
    ) {}

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    @PostConstruct
    public void init() {
        refresh();
    }

    /**
     * Rescan the python/ directory for dataset manifests. Safe to call at any time.
     */
    public void refresh() {
        Path pythonDir = findPythonDir();
        if (pythonDir == null) {
            log.warn("Could not locate python/ directory — no datasets discovered");
            return;
        }

        List<DatasetDescriptor> discovered = new ArrayList<>();

        try (DirectoryStream<Path> dirs = Files.newDirectoryStream(pythonDir, "*_example")) {
            for (Path dir : dirs) {
                Path manifest = dir.resolve(MANIFEST_FILENAME);
                if (Files.isRegularFile(manifest)) {
                    try {
                        DatasetDescriptor desc = mapper.readValue(manifest.toFile(), DatasetDescriptor.class);
                        discovered.add(desc);
                        log.info("Discovered dataset: {} ({})", desc.name(), manifest);
                    } catch (IOException e) {
                        log.warn("Failed to parse {}: {}", manifest, e.getMessage());
                    }
                }
            }
        } catch (IOException e) {
            log.warn("Error scanning python/ directory: {}", e.getMessage());
        }

        // Sort by id for stable ordering
        discovered.sort(Comparator.comparing(DatasetDescriptor::id));

        datasets.clear();
        datasets.addAll(discovered);

        log.info("Dataset discovery complete: {} dataset(s) found", datasets.size());
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    public List<DatasetDescriptor> getDatasets() {
        return Collections.unmodifiableList(datasets);
    }

    public Optional<DatasetDescriptor> getDataset(String id) {
        return datasets.stream().filter(d -> d.id().equals(id)).findFirst();
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private Path findPythonDir() {
        // Try CWD first
        String cwd = System.getProperty("user.dir");
        Path candidate = Path.of(cwd, "python");
        if (Files.isDirectory(candidate)) {
            return candidate;
        }

        // Try well-known location
        Path home = Path.of(System.getProperty("user.home"));
        candidate = home.resolve("code/warp_experiments/feature_store/python");
        if (Files.isDirectory(candidate)) {
            return candidate;
        }

        return null;
    }
}
