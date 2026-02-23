package com.platform.featurestore.ui;

import com.platform.featurestore.proto.FeatureVector;
import com.platform.featurestore.proto.ViewSchema;
import com.platform.featurestore.service.OnlineServingService;
import com.platform.featurestore.store.online.RocksDBFeatureStore;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.html.H3;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.notification.NotificationVariant;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.textfield.IntegerField;
import com.vaadin.flow.component.textfield.TextField;
import com.vaadin.flow.router.PageTitle;
import com.vaadin.flow.router.Route;

import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Route(value = "lookup", layout = MainLayout.class)
@PageTitle("Feature Lookup | Feature Store")
public class FeatureLookupView extends VerticalLayout {

    private static final DateTimeFormatter FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS")
            .withZone(ZoneId.systemDefault());

    private final OnlineServingService servingService;
    private final RocksDBFeatureStore rocksStore;

    private final VerticalLayout resultArea = new VerticalLayout();

    public FeatureLookupView(OnlineServingService servingService, RocksDBFeatureStore rocksStore) {
        this.servingService = servingService;
        this.rocksStore = rocksStore;
        setPadding(true);
        setSpacing(true);

        add(new H3("Feature Value Lookup"));

        // Form
        TextField viewNameField = new TextField("View Name");
        viewNameField.setPlaceholder("e.g. merchant_fraud_risk_v1");
        IntegerField viewVersionField = new IntegerField("View Version");
        viewVersionField.setValue(1);
        viewVersionField.setMin(1);
        TextField entityIdField = new TextField("Entity ID");
        entityIdField.setPlaceholder("e.g. merchant_001");

        Button lookupBtn = new Button("Lookup");
        lookupBtn.addThemeVariants(ButtonVariant.LUMO_PRIMARY);

        HorizontalLayout form = new HorizontalLayout(viewNameField, viewVersionField, entityIdField, lookupBtn);
        form.setDefaultVerticalComponentAlignment(Alignment.BASELINE);
        add(form);

        resultArea.setPadding(false);
        add(resultArea);

        lookupBtn.addClickListener(e -> {
            String viewName = viewNameField.getValue();
            int viewVersion = viewVersionField.getValue() != null ? viewVersionField.getValue() : 1;
            String entityId = entityIdField.getValue();

            if (viewName.isBlank() || entityId.isBlank()) {
                Notification.show("View name and entity ID are required", 3000, Notification.Position.BOTTOM_START);
                return;
            }

            performLookup(viewName, viewVersion, entityId);
        });
    }

    private void performLookup(String viewName, int viewVersion, String entityId) {
        resultArea.removeAll();

        OnlineServingService.ServingResult result = servingService.getFeatureVector(viewName, viewVersion, entityId);

        if (result.vector() == null) {
            resultArea.add(new Span("No vector found. Source: " + result.source()));
            if (!result.warnings().isEmpty()) {
                result.warnings().forEach(w -> resultArea.add(new Span("⚠ " + w)));
            }
            return;
        }

        FeatureVector fv = result.vector();

        // Metadata
        VerticalLayout meta = new VerticalLayout();
        meta.setSpacing(false);
        meta.setPadding(false);
        meta.add(new Span("Source: " + result.source() + " | Latency: " + result.latencyUs() + " µs"));
        meta.add(new Span("Schema Hash: " + fv.getSchemaHash()
                + " | Served At: " + FMT.format(Instant.ofEpochMilli(fv.getServedAtMs()))));
        meta.add(new Span("Entity: " + fv.getEntityType() + "/" + fv.getEntityId()
                + " | View: " + fv.getViewName() + " v" + fv.getViewVersion()));
        resultArea.add(meta);

        if (!result.warnings().isEmpty()) {
            result.warnings().forEach(w -> {
                Span warn = new Span("⚠ " + w);
                warn.getStyle().set("color", "var(--fs-warning)");
                resultArea.add(warn);
            });
        }

        // Try to get schema for feature names
        List<String> featureNames;
        try {
            Optional<ViewSchema> schema = rocksStore.getSchema(viewName, viewVersion);
            featureNames = schema.map(s -> s.getFeatureNamesList().stream().toList())
                    .orElse(null);
        } catch (Exception e) {
            featureNames = null;
        }

        // Values grid
        Grid<FeatureValueRow> grid = new Grid<>();
        grid.addColumn(FeatureValueRow::index).setHeader("#").setFlexGrow(0).setWidth("50px");
        grid.addColumn(FeatureValueRow::name).setHeader("Feature").setFlexGrow(1);
        grid.addColumn(r -> String.format("%.6f", r.value())).setHeader("Value").setFlexGrow(0).setWidth("150px");
        grid.addColumn(r -> {
            if (r.ageMs() < 0) return "unknown";
            long secs = r.ageMs() / 1000;
            if (secs < 60) return secs + "s";
            if (secs < 3600) return (secs / 60) + "m " + (secs % 60) + "s";
            return (secs / 3600) + "h " + ((secs % 3600) / 60) + "m";
        }).setHeader("Age").setFlexGrow(0).setWidth("100px");
        grid.addColumn(r -> r.isDefault() ? "YES" : "")
                .setHeader("Default?").setFlexGrow(0).setWidth("80px");

        List<FeatureValueRow> rows = new ArrayList<>();
        for (int i = 0; i < fv.getValuesCount(); i++) {
            String name = (featureNames != null && i < featureNames.size()) ? featureNames.get(i) : "feature_" + i;
            long ageMs = i < fv.getValueAgesMsCount() ? fv.getValueAgesMs(i) : -1;
            boolean isDefault = i < fv.getIsDefaultMaskCount() && fv.getIsDefaultMask(i);
            rows.add(new FeatureValueRow(i, name, fv.getValues(i), ageMs, isDefault));
        }

        grid.setItems(rows);
        grid.setWidthFull();
        grid.setAllRowsVisible(true);

        resultArea.add(new H3("Feature Values (" + fv.getValuesCount() + ")"));
        resultArea.add(grid);
    }

    private record FeatureValueRow(int index, String name, double value, long ageMs, boolean isDefault) {}
}
