package com.platform.featurestore.ui;

import com.platform.featurestore.domain.EntityDefinition;
import com.platform.featurestore.domain.FeatureDefinition;
import com.platform.featurestore.service.FeatureRegistryService;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.dialog.Dialog;
import com.vaadin.flow.component.formlayout.FormLayout;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.html.H3;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.notification.NotificationVariant;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.select.Select;
import com.vaadin.flow.component.textfield.IntegerField;
import com.vaadin.flow.component.textfield.TextField;
import com.vaadin.flow.router.PageTitle;
import com.vaadin.flow.router.Route;

import java.util.List;

@Route(value = "features", layout = MainLayout.class)
@PageTitle("Features | Feature Store")
public class FeatureListView extends VerticalLayout {

    private final FeatureRegistryService registryService;
    private final Grid<FeatureDefinition> grid = new Grid<>();

    private Select<EntityDefinition> entityFilter;
    private Select<String> statusFilter;

    public FeatureListView(FeatureRegistryService registryService) {
        this.registryService = registryService;
        setPadding(true);
        setSpacing(true);

        // Header with filters
        Button createBtn = new Button("Create Feature", e -> openCreateDialog());
        createBtn.addThemeVariants(ButtonVariant.LUMO_PRIMARY);

        entityFilter = new Select<>();
        entityFilter.setLabel("Entity");
        entityFilter.setPlaceholder("All entities");
        entityFilter.setItemLabelGenerator(e -> e != null ? e.getName() : "All entities");
        List<EntityDefinition> entities = registryService.listEntities();
        entityFilter.setItems(entities);
        entityFilter.setEmptySelectionAllowed(true);
        entityFilter.addValueChangeListener(e -> refreshGrid());

        statusFilter = new Select<>();
        statusFilter.setLabel("Status");
        statusFilter.setItems("ALL", "ACTIVE", "DEPRECATED");
        statusFilter.setValue("ALL");
        statusFilter.addValueChangeListener(e -> refreshGrid());

        HorizontalLayout header = new HorizontalLayout(new H3("Features"), entityFilter, statusFilter, createBtn);
        header.setDefaultVerticalComponentAlignment(Alignment.BASELINE);
        add(header);

        // Grid
        grid.addColumn(FeatureDefinition::getName).setHeader("Name").setSortable(true).setFlexGrow(1);
        grid.addColumn(f -> f.getEntity() != null ? f.getEntity().getName() : "—")
                .setHeader("Entity").setFlexGrow(0).setWidth("150px");
        grid.addColumn(FeatureDefinition::getDtype).setHeader("Type").setFlexGrow(0).setWidth("100px");
        grid.addComponentColumn(f -> {
            Span badge = new Span(f.getStatus());
            badge.addClassName("status-badge");
            badge.addClassName(f.getStatus().toLowerCase());
            return badge;
        }).setHeader("Status").setFlexGrow(0).setWidth("120px");
        grid.addColumn(FeatureDefinition::getOwner).setHeader("Owner").setFlexGrow(0).setWidth("120px");
        grid.addColumn(FeatureDefinition::getVersion).setHeader("Ver").setFlexGrow(0).setWidth("60px");
        grid.addComponentColumn(this::createActions).setHeader("Actions").setFlexGrow(0).setWidth("200px");

        grid.setWidthFull();
        grid.setHeight("600px");

        refreshGrid();
        add(grid);
    }

    private HorizontalLayout createActions(FeatureDefinition feature) {
        Button edit = new Button("Edit", e -> openEditDialog(feature));
        edit.addThemeVariants(ButtonVariant.LUMO_SMALL);

        Button deprecate = new Button("Deprecate", e -> openDeprecateDialog(feature));
        deprecate.addThemeVariants(ButtonVariant.LUMO_SMALL, ButtonVariant.LUMO_ERROR);
        deprecate.setEnabled("ACTIVE".equals(feature.getStatus()));

        return new HorizontalLayout(edit, deprecate);
    }

    private void openCreateDialog() {
        Dialog dialog = new Dialog();
        dialog.setHeaderTitle("Create Feature");

        Select<EntityDefinition> entitySelect = new Select<>();
        entitySelect.setLabel("Entity");
        entitySelect.setItemLabelGenerator(EntityDefinition::getName);
        entitySelect.setItems(registryService.listEntities());
        entitySelect.setRequiredIndicatorVisible(true);

        TextField nameField = new TextField("Name");
        nameField.setRequired(true);
        Select<String> dtypeField = new Select<>();
        dtypeField.setLabel("Data Type");
        dtypeField.setItems("FLOAT64", "FLOAT32", "INT64", "INT32", "STRING", "BOOL", "BYTES");
        dtypeField.setValue("FLOAT64");

        TextField descField = new TextField("Description");
        TextField ownerField = new TextField("Owner");
        TextField pipelineField = new TextField("Source Pipeline");
        Select<String> freqField = new Select<>();
        freqField.setLabel("Update Frequency");
        freqField.setItems("REALTIME", "HOURLY", "DAILY", "WEEKLY", "BATCH");
        IntegerField maxAgeField = new IntegerField("Max Age (seconds)");
        TextField defaultField = new TextField("Default Value");

        FormLayout form = new FormLayout(entitySelect, nameField, dtypeField, descField,
                ownerField, pipelineField, freqField, maxAgeField, defaultField);
        dialog.add(form);

        Button save = new Button("Create", e -> {
            try {
                if (entitySelect.getValue() == null) {
                    Notification.show("Select an entity", 3000, Notification.Position.BOTTOM_START);
                    return;
                }
                FeatureDefinition feature = new FeatureDefinition();
                feature.setName(nameField.getValue());
                feature.setEntity(entitySelect.getValue());
                feature.setDtype(dtypeField.getValue());
                feature.setDescription(descField.getValue());
                feature.setOwner(ownerField.getValue());
                feature.setSourcePipeline(pipelineField.getValue());
                feature.setUpdateFrequency(freqField.getValue());
                feature.setMaxAgeSeconds(maxAgeField.getValue());
                feature.setDefaultValue(defaultField.getValue());
                registryService.createFeature(feature);
                dialog.close();
                refreshGrid();
                Notification.show("Feature created", 3000, Notification.Position.BOTTOM_START)
                        .addThemeVariants(NotificationVariant.LUMO_SUCCESS);
            } catch (Exception ex) {
                Notification.show("Error: " + ex.getMessage(), 5000, Notification.Position.BOTTOM_START)
                        .addThemeVariants(NotificationVariant.LUMO_ERROR);
            }
        });
        save.addThemeVariants(ButtonVariant.LUMO_PRIMARY);

        Button cancel = new Button("Cancel", e -> dialog.close());
        dialog.getFooter().add(cancel, save);
        dialog.open();
    }

    private void openEditDialog(FeatureDefinition feature) {
        Dialog dialog = new Dialog();
        dialog.setHeaderTitle("Edit Feature: " + feature.getName());

        TextField descField = new TextField("Description");
        descField.setValue(feature.getDescription() != null ? feature.getDescription() : "");
        TextField ownerField = new TextField("Owner");
        ownerField.setValue(feature.getOwner() != null ? feature.getOwner() : "");
        Select<String> freqField = new Select<>();
        freqField.setLabel("Update Frequency");
        freqField.setItems("REALTIME", "HOURLY", "DAILY", "WEEKLY", "BATCH");
        if (feature.getUpdateFrequency() != null) freqField.setValue(feature.getUpdateFrequency());
        IntegerField maxAgeField = new IntegerField("Max Age (seconds)");
        if (feature.getMaxAgeSeconds() != null) maxAgeField.setValue(feature.getMaxAgeSeconds());
        TextField defaultField = new TextField("Default Value");
        defaultField.setValue(feature.getDefaultValue() != null ? feature.getDefaultValue() : "");

        FormLayout form = new FormLayout(descField, ownerField, freqField, maxAgeField, defaultField);
        dialog.add(form);

        Button save = new Button("Save", e -> {
            try {
                registryService.updateFeature(feature.getId(),
                        descField.getValue(), ownerField.getValue(),
                        freqField.getValue(), maxAgeField.getValue(),
                        defaultField.getValue());
                dialog.close();
                refreshGrid();
                Notification.show("Feature updated", 3000, Notification.Position.BOTTOM_START)
                        .addThemeVariants(NotificationVariant.LUMO_SUCCESS);
            } catch (Exception ex) {
                Notification.show("Error: " + ex.getMessage(), 5000, Notification.Position.BOTTOM_START)
                        .addThemeVariants(NotificationVariant.LUMO_ERROR);
            }
        });
        save.addThemeVariants(ButtonVariant.LUMO_PRIMARY);

        Button cancel = new Button("Cancel", e -> dialog.close());
        dialog.getFooter().add(cancel, save);
        dialog.open();
    }

    private void openDeprecateDialog(FeatureDefinition feature) {
        Dialog dialog = new Dialog();
        dialog.setHeaderTitle("Deprecate Feature: " + feature.getName());

        TextField messageField = new TextField("Deprecation message");
        messageField.setWidthFull();
        dialog.add(messageField);

        Button confirm = new Button("Deprecate", e -> {
            try {
                registryService.deprecateFeature(feature.getId(), messageField.getValue());
                dialog.close();
                refreshGrid();
                Notification.show("Feature deprecated", 3000, Notification.Position.BOTTOM_START)
                        .addThemeVariants(NotificationVariant.LUMO_SUCCESS);
            } catch (Exception ex) {
                Notification.show("Error: " + ex.getMessage(), 5000, Notification.Position.BOTTOM_START)
                        .addThemeVariants(NotificationVariant.LUMO_ERROR);
            }
        });
        confirm.addThemeVariants(ButtonVariant.LUMO_PRIMARY, ButtonVariant.LUMO_ERROR);

        Button cancel = new Button("Cancel", e -> dialog.close());
        dialog.getFooter().add(cancel, confirm);
        dialog.open();
    }

    private void refreshGrid() {
        List<FeatureDefinition> features;
        EntityDefinition selectedEntity = entityFilter.getValue();

        if (selectedEntity != null) {
            features = registryService.listFeaturesByEntity(selectedEntity.getId());
        } else {
            features = registryService.listAllFeatures();
        }

        String status = statusFilter.getValue();
        if (status != null && !"ALL".equals(status)) {
            features = features.stream()
                    .filter(f -> status.equals(f.getStatus()))
                    .toList();
        }

        grid.setItems(features);
    }
}
