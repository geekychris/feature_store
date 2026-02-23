package com.platform.featurestore.ui;

import com.platform.featurestore.service.DatasetDiscoveryService;
import com.platform.featurestore.service.DatasetDiscoveryService.DatasetDescriptor;
import com.platform.featurestore.service.DatasetDiscoveryService.ParamDescriptor;
import com.platform.featurestore.service.TrainingExecutionService;
import com.platform.featurestore.service.TrainingExecutionService.*;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.vaadin.flow.component.AttachEvent;
import com.vaadin.flow.component.Component;
import com.vaadin.flow.component.HasValue;
import com.vaadin.flow.component.UI;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.checkbox.Checkbox;
import com.vaadin.flow.component.combobox.ComboBox;
import com.vaadin.flow.component.details.Details;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.html.Div;
import com.vaadin.flow.component.html.H3;
import com.vaadin.flow.component.html.Paragraph;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.notification.NotificationVariant;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.textfield.IntegerField;
import com.vaadin.flow.component.textfield.NumberField;
import com.vaadin.flow.router.PageTitle;
import com.vaadin.flow.router.Route;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.*;

@Route(value = "training", layout = MainLayout.class)
@PageTitle("Training | Feature Store")
public class TrainingView extends VerticalLayout {

    private static final DateTimeFormatter FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
            .withZone(ZoneId.systemDefault());
    private static final ObjectMapper jsonMapper = new ObjectMapper();

    private final TrainingExecutionService trainingService;
    private final DatasetDiscoveryService discoveryService;

    // --- UI components ---
    private final Div logArea = new Div();
    private final Span statusLabel = new Span();
    private final Button runBtn = new Button("Run Training");
    private final Button importBtn = new Button("Import Dataset");
    private final Button cancelBtn = new Button("Cancel");
    private final Button refreshBtn = new Button("Refresh Datasets");
    private final VerticalLayout resultsPanel = new VerticalLayout();
    private final Grid<TrainingRun> historyGrid = new Grid<>();

    // Dataset selector
    private final ComboBox<DatasetDescriptor> datasetSelector = new ComboBox<>("Dataset");
    private final Paragraph datasetDescription = new Paragraph();

    // Dynamic per-dataset parameter fields (built from manifest)
    private final HorizontalLayout datasetParamsLayout = new HorizontalLayout();
    private final Map<String, HasValue<?, ?>> paramFields = new LinkedHashMap<>();

    // Shared controls
    private final Checkbox standaloneCheck = new Checkbox("Standalone Mode", true);
    private final Checkbox noSearchCheck = new Checkbox("Skip Hyperparameter Search", true);

    // Data explorer / inference controls
    private final Button previewBtn = new Button("Preview Dataset");
    private final Button exportBtn = new Button("Export");
    private final ComboBox<String> exportFormatCombo = new ComboBox<>("Format");
    private final IntegerField previewRowsField = new IntegerField("Preview Rows");
    private final Details previewDetails = new Details();
    private final VerticalLayout previewContent = new VerticalLayout();
    private final Button loadSamplesBtn = new Button("Load Samples");
    private final Button scoreSamplesBtn = new Button("Score Samples");
    private final IntegerField inferSamplesField = new IntegerField("Samples");
    private final VerticalLayout samplesPanel = new VerticalLayout();
    private final VerticalLayout scoredPanel = new VerticalLayout();
    private final VerticalLayout inferenceSummaryPanel = new VerticalLayout();

    // Editable sample data (populated by Load Samples, read by Score)
    private List<String> sampleColumns = new ArrayList<>();
    private List<String[]> sampleRows = new ArrayList<>();

    // Hyperparameter tuning fields
    private final IntegerField maxDepthField = new IntegerField("max_depth");
    private final NumberField learningRateField = new NumberField("learning_rate");
    private final IntegerField nEstimatorsField = new IntegerField("n_estimators");
    private final IntegerField minChildWeightField = new IntegerField("min_child_weight");
    private final NumberField subsampleField = new NumberField("subsample");
    private final NumberField colsampleField = new NumberField("colsample_bytree");
    private final NumberField gammaField = new NumberField("gamma");
    private final NumberField regAlphaField = new NumberField("reg_alpha");
    private final NumberField regLambdaField = new NumberField("reg_lambda");
    private final Details hyperparamDetails = new Details();

    private UI ui;

    public TrainingView(TrainingExecutionService trainingService,
                        DatasetDiscoveryService discoveryService) {
        this.trainingService = trainingService;
        this.discoveryService = discoveryService;
        setPadding(true);
        setSpacing(true);

        add(new H3("Model Training"));

        // --- Dataset selector ---
        List<DatasetDescriptor> datasets = discoveryService.getDatasets();
        datasetSelector.setItems(datasets);
        datasetSelector.setItemLabelGenerator(DatasetDescriptor::name);
        datasetSelector.setWidthFull();
        datasetSelector.addClassName("dataset-selector");
        datasetDescription.addClassName("dataset-description");
        datasetSelector.addValueChangeListener(e -> onDatasetChanged(e.getValue()));
        if (!datasets.isEmpty()) {
            datasetSelector.setValue(datasets.getFirst());
        }
        refreshBtn.addThemeVariants(ButtonVariant.LUMO_SMALL, ButtonVariant.LUMO_TERTIARY);
        refreshBtn.addClickListener(e -> onRefreshDatasets());
        HorizontalLayout selectorRow = new HorizontalLayout(datasetSelector, refreshBtn);
        selectorRow.setWidthFull();
        selectorRow.setDefaultVerticalComponentAlignment(Alignment.BASELINE);
        selectorRow.expand(datasetSelector);
        add(selectorRow, datasetDescription);

        // --- Per-dataset parameters (populated dynamically) ---
        datasetParamsLayout.setDefaultVerticalComponentAlignment(Alignment.BASELINE);
        add(datasetParamsLayout);

        // --- Shared options ---
        standaloneCheck.setTooltipText("Run without connecting to the feature store backend");
        noSearchCheck.setTooltipText("Use default hyperparameters instead of grid search");
        HorizontalLayout sharedOpts = new HorizontalLayout(standaloneCheck, noSearchCheck);
        sharedOpts.setDefaultVerticalComponentAlignment(Alignment.BASELINE);
        add(sharedOpts);

        // --- Hyperparameter tuning (collapsible) ---
        buildHyperparamPanel();
        add(hyperparamDetails);

        // --- Action buttons ---
        runBtn.addThemeVariants(ButtonVariant.LUMO_PRIMARY);
        importBtn.addThemeVariants(ButtonVariant.LUMO_CONTRAST);
        importBtn.addClassName("import-btn");
        cancelBtn.addThemeVariants(ButtonVariant.LUMO_ERROR);
        cancelBtn.setEnabled(false);

        updateStatus();

        HorizontalLayout controls = new HorizontalLayout(runBtn, importBtn, cancelBtn, statusLabel);
        controls.setDefaultVerticalComponentAlignment(Alignment.CENTER);
        add(controls);

        // --- Data Explorer ---
        add(new H3("Data Explorer"));
        previewRowsField.setValue(50);
        previewRowsField.setMin(5);
        previewRowsField.setMax(500);
        previewRowsField.setStep(10);
        previewRowsField.setWidth("100px");
        previewBtn.addThemeVariants(ButtonVariant.LUMO_SMALL);
        previewBtn.addClickListener(e -> onPreviewDataset());

        exportFormatCombo.setItems("parquet", "iceberg");
        exportFormatCombo.setValue("parquet");
        exportFormatCombo.setWidth("130px");
        exportBtn.addThemeVariants(ButtonVariant.LUMO_SMALL, ButtonVariant.LUMO_CONTRAST);
        exportBtn.addClickListener(e -> onExportDataset());

        HorizontalLayout explorerRow = new HorizontalLayout(
                previewBtn, previewRowsField, exportFormatCombo, exportBtn);
        explorerRow.setDefaultVerticalComponentAlignment(Alignment.BASELINE);
        add(explorerRow);

        previewContent.setPadding(false);
        previewContent.setSpacing(false);
        previewDetails.setSummaryText("Dataset Preview");
        previewDetails.addContent(previewContent);
        previewDetails.setOpened(false);
        previewDetails.setWidthFull();
        previewDetails.setVisible(false);
        add(previewDetails);

        // --- Inference ---
        add(new H3("Inference"));
        inferSamplesField.setValue(50);
        inferSamplesField.setMin(5);
        inferSamplesField.setMax(1000);
        inferSamplesField.setStep(10);
        inferSamplesField.setWidth("100px");
        loadSamplesBtn.addThemeVariants(ButtonVariant.LUMO_SMALL);
        loadSamplesBtn.addClickListener(e -> onLoadSamples());
        scoreSamplesBtn.addThemeVariants(ButtonVariant.LUMO_SMALL, ButtonVariant.LUMO_SUCCESS);
        scoreSamplesBtn.addClickListener(e -> onScoreSamples());

        HorizontalLayout inferRow = new HorizontalLayout(loadSamplesBtn, scoreSamplesBtn, inferSamplesField);
        inferRow.setDefaultVerticalComponentAlignment(Alignment.BASELINE);
        add(inferRow);

        // Full-width stacked samples and scored results
        samplesPanel.setPadding(false);
        samplesPanel.setSpacing(false);
        samplesPanel.setVisible(false);
        samplesPanel.setWidthFull();
        scoredPanel.setPadding(false);
        scoredPanel.setSpacing(false);
        scoredPanel.setVisible(false);
        scoredPanel.setWidthFull();
        add(samplesPanel, scoredPanel);

        inferenceSummaryPanel.setPadding(false);
        inferenceSummaryPanel.setVisible(false);
        add(inferenceSummaryPanel);

        // --- Log output ---
        add(new H3("Output"));
        logArea.addClassName("log-area");
        logArea.setWidthFull();
        logArea.getStyle().set("min-height", "300px").set("max-height", "500px");
        logArea.setText("Ready...");
        add(logArea);

        // --- Results ---
        resultsPanel.setPadding(false);
        resultsPanel.setVisible(false);
        add(resultsPanel);

        // --- Run history ---
        add(new H3("Run History"));
        buildHistoryGrid();
        add(historyGrid);

        // --- Event handlers ---
        runBtn.addClickListener(e -> onRunTraining());
        importBtn.addClickListener(e -> onImportDataset());
        cancelBtn.addClickListener(e -> {
            trainingService.cancelTraining();
            appendLog(">>> Cancelled by user");
            updateStatus();
        });
    }

    // -----------------------------------------------------------------------
    // Initialisation helpers
    // -----------------------------------------------------------------------

    private void buildHyperparamPanel() {
        maxDepthField.setValue(6);          maxDepthField.setMin(1); maxDepthField.setMax(15);
        learningRateField.setValue(0.1);    learningRateField.setMin(0.001); learningRateField.setMax(1.0); learningRateField.setStep(0.01);
        nEstimatorsField.setValue(200);     nEstimatorsField.setMin(10); nEstimatorsField.setMax(5000); nEstimatorsField.setStep(10);
        minChildWeightField.setValue(1);    minChildWeightField.setMin(1); minChildWeightField.setMax(20);
        subsampleField.setValue(0.8);       subsampleField.setMin(0.1); subsampleField.setMax(1.0); subsampleField.setStep(0.05);
        colsampleField.setValue(0.8);       colsampleField.setMin(0.1); colsampleField.setMax(1.0); colsampleField.setStep(0.05);
        gammaField.setValue(0.0);           gammaField.setMin(0.0); gammaField.setMax(10.0); gammaField.setStep(0.1);
        regAlphaField.setValue(0.0);        regAlphaField.setMin(0.0); regAlphaField.setMax(10.0); regAlphaField.setStep(0.1);
        regLambdaField.setValue(1.0);       regLambdaField.setMin(0.0); regLambdaField.setMax(10.0); regLambdaField.setStep(0.1);

        HorizontalLayout row1 = new HorizontalLayout(maxDepthField, learningRateField, nEstimatorsField, minChildWeightField);
        HorizontalLayout row2 = new HorizontalLayout(subsampleField, colsampleField, gammaField, regAlphaField, regLambdaField);
        row1.setDefaultVerticalComponentAlignment(Alignment.BASELINE);
        row2.setDefaultVerticalComponentAlignment(Alignment.BASELINE);

        VerticalLayout content = new VerticalLayout(row1, row2);
        content.setSpacing(true);
        content.setPadding(false);
        content.addClassName("hyperparam-panel");

        hyperparamDetails.setSummaryText("Hyperparameter Tuning (XGBoost)");
        hyperparamDetails.addContent(content);
        hyperparamDetails.setOpened(false);
        hyperparamDetails.addClassName("hyperparam-details");
    }

    private void buildHistoryGrid() {
        historyGrid.addColumn(TrainingRun::id).setHeader("Run ID").setFlexGrow(0).setWidth("100px");
        historyGrid.addColumn(r -> r.startedAt() != null ? FMT.format(r.startedAt()) : "")
                .setHeader("Started").setFlexGrow(0).setWidth("180px");
        historyGrid.addColumn(r -> r.completedAt() != null ? FMT.format(r.completedAt()) : "")
                .setHeader("Completed").setFlexGrow(0).setWidth("180px");
        historyGrid.addComponentColumn(r -> {
            Span badge = new Span(r.status().name());
            badge.addClassName("status-badge");
            badge.addClassName(r.status().name().toLowerCase());
            return badge;
        }).setHeader("Status").setFlexGrow(0).setWidth("120px");
        historyGrid.addColumn(this::formatPrimaryMetric).setHeader("Metric").setFlexGrow(0).setWidth("120px");
        historyGrid.addColumn(r -> r.config().toString()).setHeader("Config").setFlexGrow(1);
        historyGrid.setHeight("250px");
        historyGrid.setWidthFull();
        refreshHistory();
        historyGrid.addItemClickListener(e -> showRunLogs(e.getItem()));
    }

    // -----------------------------------------------------------------------
    // Dataset switching (fully dynamic from manifest params)
    // -----------------------------------------------------------------------

    private void onDatasetChanged(DatasetDescriptor ds) {
        if (ds == null) return;
        datasetDescription.setText(ds.description());
        datasetParamsLayout.removeAll();
        paramFields.clear();

        for (ParamDescriptor param : ds.params()) {
            if ("integer".equals(param.type())) {
                IntegerField field = new IntegerField(param.label());
                if (param.defaultValue() != null) field.setValue(param.defaultValue().intValue());
                if (param.min() != null) field.setMin(param.min().intValue());
                if (param.max() != null) field.setMax(param.max().intValue());
                if (param.step() != null) field.setStep(param.step().intValue());
                datasetParamsLayout.add(field);
                paramFields.put(param.name(), field);
            } else {
                NumberField field = new NumberField(param.label());
                if (param.defaultValue() != null) field.setValue(param.defaultValue().doubleValue());
                if (param.min() != null) field.setMin(param.min().doubleValue());
                if (param.max() != null) field.setMax(param.max().doubleValue());
                if (param.step() != null) field.setStep(param.step().doubleValue());
                datasetParamsLayout.add(field);
                paramFields.put(param.name(), field);
            }
        }
    }

    private void onRefreshDatasets() {
        discoveryService.refresh();
        List<DatasetDescriptor> datasets = discoveryService.getDatasets();
        datasetSelector.setItems(datasets);
        if (!datasets.isEmpty()) {
            datasetSelector.setValue(datasets.getFirst());
        }
        Notification.show(datasets.size() + " dataset(s) discovered", 3000, Notification.Position.BOTTOM_START);
    }

    // -----------------------------------------------------------------------
    // Collect current parameter values
    // -----------------------------------------------------------------------

    private Map<String, Object> collectDatasetParams() {
        Map<String, Object> params = new LinkedHashMap<>();
        for (var entry : paramFields.entrySet()) {
            Object val = entry.getValue().getValue();
            if (val != null) params.put(entry.getKey(), val);
        }
        return params;
    }

    private Map<String, Object> collectHyperparams() {
        Map<String, Object> hp = new LinkedHashMap<>();
        hp.put("max_depth", maxDepthField.getValue());
        hp.put("learning_rate", learningRateField.getValue());
        hp.put("n_estimators", nEstimatorsField.getValue());
        hp.put("min_child_weight", minChildWeightField.getValue());
        hp.put("subsample", subsampleField.getValue());
        hp.put("colsample_bytree", colsampleField.getValue());
        hp.put("gamma", gammaField.getValue());
        hp.put("reg_alpha", regAlphaField.getValue());
        hp.put("reg_lambda", regLambdaField.getValue());
        return hp;
    }

    // -----------------------------------------------------------------------
    // Actions
    // -----------------------------------------------------------------------

    private void onRunTraining() {
        try {
            logArea.setText("");
            resultsPanel.setVisible(false);
            resultsPanel.removeAll();

            DatasetDescriptor ds = datasetSelector.getValue();
            if (ds == null) {
                Notification.show("Select a dataset first", 3000, Notification.Position.BOTTOM_START)
                        .addThemeVariants(NotificationVariant.LUMO_ERROR);
                return;
            }

            TrainingConfig config = new TrainingConfig(
                    ds,
                    standaloneCheck.getValue(),
                    noSearchCheck.getValue(),
                    collectDatasetParams(),
                    collectHyperparams()
            );

            String runId = trainingService.startTraining(config, logLine -> {
                if (ui != null) {
                    ui.access(() -> { appendLog(logLine); ui.push(); });
                }
            });

            Notification.show("Training started: " + runId, 3000, Notification.Position.BOTTOM_START)
                    .addThemeVariants(NotificationVariant.LUMO_SUCCESS);
            updateStatus();
            pollForCompletion();

        } catch (IllegalStateException ex) {
            Notification.show(ex.getMessage(), 5000, Notification.Position.BOTTOM_START)
                    .addThemeVariants(NotificationVariant.LUMO_ERROR);
        }
    }

    private void onImportDataset() {
        try {
            logArea.setText("");
            resultsPanel.setVisible(false);
            resultsPanel.removeAll();

            DatasetDescriptor ds = datasetSelector.getValue();
            if (ds == null) {
                Notification.show("Select a dataset first", 3000, Notification.Position.BOTTOM_START)
                        .addThemeVariants(NotificationVariant.LUMO_ERROR);
                return;
            }

            String runId = trainingService.startDatasetImport(ds, collectDatasetParams(), logLine -> {
                if (ui != null) {
                    ui.access(() -> { appendLog(logLine); ui.push(); });
                }
            });

            Notification.show("Import started: " + runId, 3000, Notification.Position.BOTTOM_START)
                    .addThemeVariants(NotificationVariant.LUMO_SUCCESS);
            updateStatus();
            pollForCompletion();

        } catch (IllegalStateException ex) {
            Notification.show(ex.getMessage(), 5000, Notification.Position.BOTTOM_START)
                    .addThemeVariants(NotificationVariant.LUMO_ERROR);
        }
    }

    // -----------------------------------------------------------------------
    // Lifecycle & polling
    // -----------------------------------------------------------------------

    @Override
    protected void onAttach(AttachEvent event) {
        super.onAttach(event);
        this.ui = event.getUI();
        updateStatus();
    }

    private void pollForCompletion() {
        if (ui != null) {
            ui.access(() -> {
                RunStatus status = trainingService.getStatus();
                updateStatus();
                if (status == RunStatus.RUNNING) {
                    ui.getPage().executeJs(
                            "setTimeout(() => $0.$server.checkStatus(), 2000)",
                            getElement()
                    );
                } else {
                    onComplete();
                }
                ui.push();
            });
        }
    }

    @com.vaadin.flow.component.ClientCallable
    public void checkStatus() {
        pollForCompletion();
    }

    private void onComplete() {
        refreshHistory();
        if (!trainingService.getRunHistory().isEmpty()) {
            TrainingRun latest = trainingService.getRunHistory().getFirst();
            showResults(latest);
        }
    }

    // -----------------------------------------------------------------------
    // Log & status helpers
    // -----------------------------------------------------------------------

    private void appendLog(String line) {
        String current = logArea.getText();
        if (current == null || "Ready...".equals(current)) {
            logArea.setText(line);
        } else {
            logArea.setText(current + "\n" + line);
        }
        logArea.getElement().executeJs("this.scrollTop = this.scrollHeight");
    }

    private void updateStatus() {
        RunStatus status = trainingService.getStatus();
        statusLabel.setText("Status: " + status.name());
        statusLabel.removeClassNames("status-active", "status-deprecated", "status-badge",
                "running", "active", "failed");

        boolean busy = status == RunStatus.RUNNING;
        runBtn.setEnabled(!busy);
        importBtn.setEnabled(!busy);
        cancelBtn.setEnabled(busy);

        statusLabel.addClassName("status-badge");
        switch (status) {
            case RUNNING   -> statusLabel.addClassName("running");
            case COMPLETED -> statusLabel.addClassName("active");
            case FAILED    -> statusLabel.addClassName("failed");
            default -> {}
        }
    }

    // -----------------------------------------------------------------------
    // Results panel (metrics driven by descriptor)
    // -----------------------------------------------------------------------

    private void showResults(TrainingRun run) {
        resultsPanel.removeAll();
        resultsPanel.setVisible(true);

        resultsPanel.add(new H3("Results — Run " + run.id()));

        Map<String, Object> results = run.results();
        if (results.isEmpty()) {
            resultsPanel.add(new Span("No parsed results available."));
            return;
        }

        VerticalLayout metrics = new VerticalLayout();
        metrics.setSpacing(false);
        metrics.setPadding(false);

        // Show metrics from the selected descriptor if available, otherwise show all numeric results
        DatasetDescriptor ds = datasetSelector.getValue();
        if (ds != null && ds.metrics() != null) {
            for (String metricKey : ds.metrics()) {
                addMetricRow(metrics, formatMetricLabel(metricKey), results.get(metricKey));
            }
        }
        // Always show extra universal metrics
        addMetricRow(metrics, "Train time (s)", results.get("training_time_sec"));

        Object gatesPassed = results.get("gates_passed");
        if (gatesPassed != null) {
            Span lbl = new Span("Quality Gates: ");
            lbl.getStyle().set("font-weight", "bold");
            Span val = new Span(Boolean.TRUE.equals(gatesPassed) ? "✅ Passed" : "❌ Failed");
            metrics.add(new HorizontalLayout(lbl, val));
        }

        resultsPanel.add(metrics);

        // Feature importance
        Object fi = results.get("feature_importance");
        if (fi instanceof Map<?, ?> fiMap && !fiMap.isEmpty()) {
            resultsPanel.add(new H3("Feature Importance"));
            Grid<Map.Entry<String, Double>> fiGrid = new Grid<>();
            fiGrid.addColumn(Map.Entry::getKey).setHeader("Feature").setSortable(true).setFlexGrow(1);
            fiGrid.addColumn(e -> String.format("%.4f", e.getValue()))
                    .setHeader("Importance").setSortable(true).setFlexGrow(0).setWidth("120px");
            @SuppressWarnings("unchecked")
            var entries = ((Map<String, Double>) fi).entrySet().stream()
                    .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                    .toList();
            fiGrid.setItems(entries);
            fiGrid.setWidthFull();
            fiGrid.setAllRowsVisible(true);
            resultsPanel.add(fiGrid);
        }
    }

    private void addMetricRow(VerticalLayout container, String label, Object value) {
        if (value != null) {
            Span lbl = new Span(label + ": ");
            lbl.getStyle().set("font-weight", "bold");
            Span val = new Span(value instanceof Number n ? String.format("%.4f", n.doubleValue()) : value.toString());
            HorizontalLayout row = new HorizontalLayout(lbl, val);
            row.setSpacing(true);
            container.add(row);
        }
    }

    /** Convert metric keys like "auc_roc" to friendly labels like "AUC-ROC". */
    private static String formatMetricLabel(String key) {
        return switch (key) {
            case "auc_roc" -> "AUC-ROC";
            case "auc_pr"  -> "AUC-PR";
            case "logloss" -> "Log-Loss";
            case "ndcg_1"  -> "NDCG@1";
            case "ndcg_5"  -> "NDCG@5";
            case "ndcg_10" -> "NDCG@10";
            case "map_score" -> "MAP";
            case "accuracy" -> "Accuracy";
            case "f1_score" -> "F1 Score";
            default -> key.replace('_', ' ');
        };
    }

    private String formatPrimaryMetric(TrainingRun r) {
        Map<String, Object> res = r.results();
        if (res == null || res.isEmpty()) return "—";

        // Try to use the selected dataset's primary metric
        DatasetDescriptor ds = datasetSelector.getValue();
        if (ds != null && ds.primaryMetric() != null) {
            Object v = res.get(ds.primaryMetric());
            if (v instanceof Number n) {
                return ds.primaryMetric() + "=" + String.format("%.4f", n.doubleValue());
            }
        }

        // Fallback: first numeric result
        for (var entry : res.entrySet()) {
            if (entry.getValue() instanceof Number n && !"training_time_sec".equals(entry.getKey())) {
                return entry.getKey() + "=" + String.format("%.4f", n.doubleValue());
            }
        }
        return "—";
    }

    // -----------------------------------------------------------------------
    // Preview
    // -----------------------------------------------------------------------

    private void onPreviewDataset() {
        try {
            DatasetDescriptor ds = datasetSelector.getValue();
            if (ds == null) {
                Notification.show("Select a dataset first", 3000, Notification.Position.BOTTOM_START)
                        .addThemeVariants(NotificationVariant.LUMO_ERROR);
                return;
            }

            logArea.setText("");
            int rows = previewRowsField.getValue() != null ? previewRowsField.getValue() : 50;

            trainingService.startDatasetPreview(ds, collectDatasetParams(), rows,
                    logLine -> {
                        if (ui != null) ui.access(() -> { appendLog(logLine); ui.push(); });
                    },
                    jsonPayload -> {
                        if (ui != null) ui.access(() -> { renderPreview(jsonPayload); ui.push(); });
                    });

            Notification.show("Generating preview...", 2000, Notification.Position.BOTTOM_START);
            updateStatus();
            pollForCompletion();

        } catch (IllegalStateException ex) {
            Notification.show(ex.getMessage(), 5000, Notification.Position.BOTTOM_START)
                    .addThemeVariants(NotificationVariant.LUMO_ERROR);
        }
    }

    private void renderPreview(String jsonPayload) {
        try {
            JsonNode root = jsonMapper.readTree(jsonPayload);
            JsonNode columns = root.get("columns");
            JsonNode data = root.get("data");
            int totalRows = root.has("total_rows") ? root.get("total_rows").asInt() : data.size();

            previewContent.removeAll();

            Grid<List<String>> grid = buildTableGrid(columns, data);
            grid.setAllRowsVisible(true);

            previewContent.add(
                    new Span("Showing " + data.size() + " of " + totalRows + " rows, " + columns.size() + " columns"),
                    grid
            );
            previewDetails.setOpened(true);
            previewDetails.setVisible(true);

        } catch (Exception e) {
            previewContent.removeAll();
            previewContent.add(new Span("Failed to parse preview: " + e.getMessage()));
            previewDetails.setOpened(true);
            previewDetails.setVisible(true);
        }
    }

    // -----------------------------------------------------------------------
    // Export
    // -----------------------------------------------------------------------

    private void onExportDataset() {
        try {
            DatasetDescriptor ds = datasetSelector.getValue();
            if (ds == null) {
                Notification.show("Select a dataset first", 3000, Notification.Position.BOTTOM_START)
                        .addThemeVariants(NotificationVariant.LUMO_ERROR);
                return;
            }

            String format = exportFormatCombo.getValue();
            if (format == null || format.isBlank()) format = "parquet";

            logArea.setText("");

            trainingService.startDatasetExport(ds, collectDatasetParams(), format,
                    logLine -> {
                        if (ui != null) ui.access(() -> { appendLog(logLine); ui.push(); });
                    });

            Notification.show("Export started (" + format + ")...", 2000, Notification.Position.BOTTOM_START)
                    .addThemeVariants(NotificationVariant.LUMO_SUCCESS);
            updateStatus();
            pollForCompletion();

        } catch (IllegalStateException ex) {
            Notification.show(ex.getMessage(), 5000, Notification.Position.BOTTOM_START)
                    .addThemeVariants(NotificationVariant.LUMO_ERROR);
        }
    }

    // -----------------------------------------------------------------------
    // Inference — two-step: Load Samples → Score Samples
    // -----------------------------------------------------------------------

    private void onLoadSamples() {
        try {
            DatasetDescriptor ds = datasetSelector.getValue();
            if (ds == null) {
                Notification.show("Select a dataset first", 3000, Notification.Position.BOTTOM_START)
                        .addThemeVariants(NotificationVariant.LUMO_ERROR);
                return;
            }

            logArea.setText("");
            samplesPanel.removeAll();
            scoredPanel.removeAll();
            scoredPanel.setVisible(false);
            inferenceSummaryPanel.removeAll();
            inferenceSummaryPanel.setVisible(false);

            int rows = inferSamplesField.getValue() != null ? inferSamplesField.getValue() : 50;

            trainingService.startDatasetPreview(ds, collectDatasetParams(), rows,
                    logLine -> {
                        if (ui != null) ui.access(() -> { appendLog(logLine); ui.push(); });
                    },
                    jsonPayload -> {
                        if (ui != null) ui.access(() -> { renderSamplesGrid(jsonPayload); ui.push(); });
                    });

            Notification.show("Loading samples...", 2000, Notification.Position.BOTTOM_START);
            updateStatus();
            pollForCompletion();

        } catch (IllegalStateException ex) {
            Notification.show(ex.getMessage(), 5000, Notification.Position.BOTTOM_START)
                    .addThemeVariants(NotificationVariant.LUMO_ERROR);
        }
    }

    private void onScoreSamples() {
        try {
            DatasetDescriptor ds = datasetSelector.getValue();
            if (ds == null) {
                Notification.show("Select a dataset first", 3000, Notification.Position.BOTTOM_START)
                        .addThemeVariants(NotificationVariant.LUMO_ERROR);
                return;
            }

            logArea.setText("");
            scoredPanel.removeAll();
            scoredPanel.setVisible(false);
            inferenceSummaryPanel.removeAll();
            inferenceSummaryPanel.setVisible(false);

            int nSamples = inferSamplesField.getValue() != null ? inferSamplesField.getValue() : 50;

            // If we have edited sample data, write to temp file
            Path inputFile = null;
            if (!sampleRows.isEmpty()) {
                inputFile = writeEditedSamplesToFile();
            }

            trainingService.startInference(ds, nSamples, inputFile,
                    logLine -> {
                        if (ui != null) ui.access(() -> { appendLog(logLine); ui.push(); });
                    },
                    sampleJson -> {
                        // Only render samples if we didn't already have edited data loaded
                        if (sampleRows.isEmpty() && ui != null) {
                            ui.access(() -> { renderSamplesGrid(sampleJson); ui.push(); });
                        }
                    },
                    inferJson -> {
                        if (ui != null) ui.access(() -> { renderScoredGrid(inferJson); ui.push(); });
                    });

            Notification.show("Scoring samples...", 2000, Notification.Position.BOTTOM_START);
            updateStatus();
            pollForCompletion();

        } catch (Exception ex) {
            Notification.show(ex.getMessage(), 5000, Notification.Position.BOTTOM_START)
                    .addThemeVariants(NotificationVariant.LUMO_ERROR);
        }
    }

    private Path writeEditedSamplesToFile() throws IOException {
        // Build JSON in the same format as __SAMPLE_JSON__
        List<List<String>> data = new ArrayList<>();
        for (String[] row : sampleRows) {
            data.add(Arrays.asList(row));
        }
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("columns", sampleColumns);
        payload.put("data", data);

        Path tempFile = Files.createTempFile("inference_samples_", ".json");
        Files.writeString(tempFile, jsonMapper.writeValueAsString(payload));
        return tempFile;
    }

    private void renderSamplesGrid(String jsonPayload) {
        try {
            JsonNode root = jsonMapper.readTree(jsonPayload);
            JsonNode columns = root.get("columns");
            JsonNode data = root.get("data");
            int totalRows = root.has("total_rows") ? root.get("total_rows").asInt() : data.size();

            // Store editable data
            sampleColumns = new ArrayList<>();
            for (int i = 0; i < columns.size(); i++) {
                sampleColumns.add(columns.get(i).asText());
            }
            sampleRows = new ArrayList<>();
            for (JsonNode rowNode : data) {
                String[] row = new String[columns.size()];
                for (int i = 0; i < rowNode.size(); i++) {
                    row[i] = rowNode.get(i).isNull() ? "" : rowNode.get(i).asText();
                }
                sampleRows.add(row);
            }

            samplesPanel.removeAll();
            samplesPanel.add(new Span("Input Samples (" + sampleRows.size()
                    + " of " + totalRows + " rows) — click cells to edit"));

            // Editable grid with TextFields
            Grid<String[]> grid = new Grid<>();
            for (int i = 0; i < sampleColumns.size(); i++) {
                final int colIdx = i;
                String colName = sampleColumns.get(colIdx);
                grid.addComponentColumn(row -> {
                    com.vaadin.flow.component.textfield.TextField tf =
                            new com.vaadin.flow.component.textfield.TextField();
                    tf.setValue(row[colIdx] != null ? row[colIdx] : "");
                    tf.setWidthFull();
                    tf.addClassName("editable-cell");
                    tf.getStyle().set("--lumo-text-field-size", "var(--lumo-size-xs)");
                    tf.addValueChangeListener(e -> row[colIdx] = e.getValue());
                    return tf;
                }).setHeader(colName).setAutoWidth(true).setResizable(true);
            }
            grid.setItems(sampleRows);
            grid.setWidthFull();
            grid.setHeight("400px");
            samplesPanel.add(grid);
            samplesPanel.setVisible(true);

        } catch (Exception e) {
            samplesPanel.removeAll();
            samplesPanel.add(new Span("Failed to load samples: " + e.getMessage()));
            samplesPanel.setVisible(true);
        }
    }

    private void renderScoredGrid(String jsonPayload) {
        try {
            JsonNode root = jsonMapper.readTree(jsonPayload);
            String type = root.has("type") ? root.get("type").asText() : "unknown";
            JsonNode summary = root.get("summary");

            // Classification/CTR: "scored" contains full features + predictions table
            JsonNode scored = root.get("scored");
            // Ranking: "results" contains per-query results
            JsonNode results = root.get("results");

            scoredPanel.removeAll();

            if (scored != null) {
                // Full table with features + predictions
                JsonNode columns = scored.get("columns");
                JsonNode data = scored.get("data");
                scoredPanel.add(new Span("Scored Results (" + type + " — " + data.size() + " rows)"));

                Grid<List<String>> grid = buildTableGrid(columns, data);
                grid.setHeight("400px");
                scoredPanel.add(grid);

            } else if (results != null && results.isArray() && !results.isEmpty()) {
                // Ranking: per-query view
                scoredPanel.add(new Span("Ranking Results (" + type + " — " + results.size() + " queries)"));
                JsonNode first = results.get(0);
                Grid<JsonNode> grid = new Grid<>();
                first.fieldNames().forEachRemaining(fieldName ->
                    grid.addColumn(node -> node.has(fieldName)
                            ? (node.get(fieldName).isNumber()
                                ? String.format("%.4f", node.get(fieldName).asDouble())
                                : node.get(fieldName).asText())
                            : "")
                        .setHeader(fieldName)
                        .setSortable(true)
                        .setAutoWidth(true)
                );
                List<JsonNode> items = new ArrayList<>();
                results.forEach(items::add);
                grid.setItems(items);
                grid.setWidthFull();
                grid.setHeight("400px");
                scoredPanel.add(grid);
            }

            scoredPanel.setVisible(true);

            // Summary below both grids
            if (summary != null) {
                inferenceSummaryPanel.removeAll();
                inferenceSummaryPanel.add(new H3("Inference Summary"));
                VerticalLayout summaryLayout = new VerticalLayout();
                summaryLayout.setSpacing(false);
                summaryLayout.setPadding(false);
                summary.fields().forEachRemaining(entry -> {
                    Span lbl = new Span(formatMetricLabel(entry.getKey()) + ": ");
                    lbl.getStyle().set("font-weight", "bold");
                    Span val = new Span(entry.getValue().isNumber()
                            ? String.format("%.4f", entry.getValue().asDouble())
                            : entry.getValue().asText());
                    summaryLayout.add(new HorizontalLayout(lbl, val));
                });
                inferenceSummaryPanel.add(summaryLayout);
                inferenceSummaryPanel.setVisible(true);
            }

        } catch (Exception e) {
            scoredPanel.removeAll();
            scoredPanel.add(new Span("Failed to parse results: " + e.getMessage()));
            scoredPanel.setVisible(true);
        }
    }

    /** Build a Grid from columns + data JSON arrays (shared by preview, samples, scored). */
    private Grid<List<String>> buildTableGrid(JsonNode columns, JsonNode data) {
        Grid<List<String>> grid = new Grid<>();
        for (int i = 0; i < columns.size(); i++) {
            final int colIdx = i;
            String colName = columns.get(i).asText();
            grid.addColumn(row -> row.size() > colIdx ? row.get(colIdx) : "")
                    .setHeader(colName)
                    .setSortable(true)
                    .setAutoWidth(true);
        }

        List<List<String>> rows = new ArrayList<>();
        for (JsonNode rowNode : data) {
            List<String> row = new ArrayList<>();
            for (JsonNode cell : rowNode) {
                row.add(cell.isNull() ? "" : cell.asText());
            }
            rows.add(row);
        }
        grid.setItems(rows);
        grid.setWidthFull();
        return grid;
    }

    // -----------------------------------------------------------------------
    // History
    // -----------------------------------------------------------------------

    private void showRunLogs(TrainingRun run) {
        logArea.setText(String.join("\n", run.logLines()));
        logArea.getElement().executeJs("this.scrollTop = this.scrollHeight");
        if (!run.results().isEmpty()) {
            showResults(run);
        }
    }

    private void refreshHistory() {
        historyGrid.setItems(trainingService.getRunHistory());
    }
}
