package com.platform.featurestore.ui;

import com.platform.featurestore.domain.EntityDefinition;
import com.platform.featurestore.domain.FeatureDefinition;
import com.platform.featurestore.domain.FeatureStatistics;
import com.platform.featurestore.service.FeatureRegistryService;
import com.vaadin.flow.component.html.H3;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.select.Select;
import com.vaadin.flow.router.PageTitle;
import com.vaadin.flow.router.Route;

import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Optional;

@Route(value = "statistics", layout = MainLayout.class)
@PageTitle("Statistics | Feature Store")
public class StatisticsView extends VerticalLayout {

    private static final DateTimeFormatter FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    private final FeatureRegistryService registryService;
    private final VerticalLayout statsContent = new VerticalLayout();

    public StatisticsView(FeatureRegistryService registryService) {
        this.registryService = registryService;
        setPadding(true);
        setSpacing(true);

        add(new H3("Feature Statistics"));

        // Selectors
        Select<EntityDefinition> entitySelect = new Select<>();
        entitySelect.setLabel("Entity");
        entitySelect.setItemLabelGenerator(EntityDefinition::getName);
        entitySelect.setItems(registryService.listEntities());

        Select<FeatureDefinition> featureSelect = new Select<>();
        featureSelect.setLabel("Feature");
        featureSelect.setItemLabelGenerator(FeatureDefinition::getName);

        entitySelect.addValueChangeListener(e -> {
            if (e.getValue() != null) {
                List<FeatureDefinition> features = registryService.listFeaturesByEntity(e.getValue().getId());
                featureSelect.setItems(features);
                featureSelect.clear();
                statsContent.removeAll();
            }
        });

        featureSelect.addValueChangeListener(e -> {
            if (e.getValue() != null) {
                showStatistics(e.getValue());
            }
        });

        HorizontalLayout selectors = new HorizontalLayout(entitySelect, featureSelect);
        selectors.setDefaultVerticalComponentAlignment(Alignment.BASELINE);
        add(selectors);

        statsContent.setPadding(false);
        add(statsContent);
    }

    private void showStatistics(FeatureDefinition feature) {
        statsContent.removeAll();

        Optional<FeatureStatistics> latestOpt = registryService.getLatestStatistics(feature.getId());

        if (latestOpt.isEmpty()) {
            statsContent.add(new Span("No statistics available for feature '" + feature.getName() + "'."));
            statsContent.add(new Span("Statistics are computed during feature materialization or training runs."));
            return;
        }

        FeatureStatistics stats = latestOpt.get();

        statsContent.add(new H3("Latest Statistics for: " + feature.getName()));

        VerticalLayout grid = new VerticalLayout();
        grid.setSpacing(false);
        grid.setPadding(false);

        grid.add(createStatRow("Computed At",
                stats.getComputedAt() != null ? stats.getComputedAt().format(FMT) : "—"));
        if (stats.getWindowStart() != null && stats.getWindowEnd() != null) {
            grid.add(createStatRow("Window",
                    stats.getWindowStart().format(FMT) + " — " + stats.getWindowEnd().format(FMT)));
        }

        grid.add(createStatRow("Count", formatLong(stats.getCount())));
        grid.add(createStatRow("Null Count", formatLong(stats.getNullCount())));
        grid.add(createStatRow("Mean", formatDouble(stats.getMean())));
        grid.add(createStatRow("Std Dev", formatDouble(stats.getStddev())));
        grid.add(createStatRow("Min", formatDouble(stats.getMinValue())));
        grid.add(createStatRow("Max", formatDouble(stats.getMaxValue())));

        statsContent.add(grid);

        // Percentiles
        statsContent.add(new H3("Percentiles"));
        VerticalLayout percGrid = new VerticalLayout();
        percGrid.setSpacing(false);
        percGrid.setPadding(false);
        percGrid.add(createStatRow("P25", formatDouble(stats.getP25())));
        percGrid.add(createStatRow("P50 (Median)", formatDouble(stats.getP50())));
        percGrid.add(createStatRow("P75", formatDouble(stats.getP75())));
        percGrid.add(createStatRow("P95", formatDouble(stats.getP95())));
        percGrid.add(createStatRow("P99", formatDouble(stats.getP99())));
        statsContent.add(percGrid);

        // Histogram bins
        if (stats.getHistogramBins() != null && !stats.getHistogramBins().isBlank()) {
            statsContent.add(new H3("Histogram Bins"));
            Span bins = new Span(stats.getHistogramBins());
            bins.getStyle().set("font-family", "monospace")
                    .set("font-size", "var(--lumo-font-size-s)")
                    .set("white-space", "pre-wrap");
            statsContent.add(bins);
        }
    }

    private HorizontalLayout createStatRow(String label, String value) {
        Span labelSpan = new Span(label + ":");
        labelSpan.getStyle().set("font-weight", "bold").set("min-width", "140px");
        Span valueSpan = new Span(value);
        HorizontalLayout row = new HorizontalLayout(labelSpan, valueSpan);
        row.setSpacing(true);
        return row;
    }

    private String formatDouble(Double val) {
        return val != null ? String.format("%.6f", val) : "—";
    }

    private String formatLong(Long val) {
        return val != null ? String.format("%,d", val) : "—";
    }
}
