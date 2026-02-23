package com.platform.featurestore.ui;

import com.platform.featurestore.domain.EntityDefinition;
import com.platform.featurestore.domain.FeatureDefinition;
import com.platform.featurestore.domain.FeatureViewDefinition;
import com.platform.featurestore.service.FeatureRegistryService;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.checkbox.Checkbox;
import com.vaadin.flow.component.confirmdialog.ConfirmDialog;
import com.vaadin.flow.component.dialog.Dialog;
import com.vaadin.flow.component.formlayout.FormLayout;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.html.H3;
import com.vaadin.flow.component.listbox.MultiSelectListBox;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.notification.NotificationVariant;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.select.Select;
import com.vaadin.flow.component.textfield.IntegerField;
import com.vaadin.flow.component.textfield.TextField;
import com.vaadin.flow.router.PageTitle;
import com.vaadin.flow.router.Route;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.UUID;

@Route(value = "feature-views", layout = MainLayout.class)
@PageTitle("Feature Views | Feature Store")
public class FeatureViewListView extends VerticalLayout {

    private final FeatureRegistryService registryService;
    private final Grid<FeatureViewDefinition> grid = new Grid<>();

    public FeatureViewListView(FeatureRegistryService registryService) {
        this.registryService = registryService;
        setPadding(true);
        setSpacing(true);

        Button createBtn = new Button("Create Feature View", e -> openCreateDialog());
        createBtn.addThemeVariants(ButtonVariant.LUMO_PRIMARY);

        HorizontalLayout header = new HorizontalLayout(new H3("Feature Views"), createBtn);
        header.setDefaultVerticalComponentAlignment(Alignment.BASELINE);
        add(header);

        grid.addColumn(FeatureViewDefinition::getName).setHeader("Name").setSortable(true).setFlexGrow(1);
        grid.addColumn(FeatureViewDefinition::getVersion).setHeader("Version").setFlexGrow(0).setWidth("80px");
        grid.addColumn(v -> v.getEntity() != null ? v.getEntity().getName() : "—")
                .setHeader("Entity").setFlexGrow(0).setWidth("150px");
        grid.addColumn(v -> v.getVectorLength() != null ? v.getVectorLength() : 0)
                .setHeader("Vec Length").setFlexGrow(0).setWidth("100px");
        grid.addColumn(v -> v.getModelName() != null ? v.getModelName() : "—")
                .setHeader("Model").setFlexGrow(0).setWidth("150px");
        grid.addColumn(FeatureViewDefinition::getStatus).setHeader("Status").setFlexGrow(0).setWidth("100px");
        grid.addComponentColumn(this::createActions).setHeader("Actions").setFlexGrow(0).setWidth("200px");

        grid.setWidthFull();
        grid.setHeight("600px");

        grid.addItemClickListener(e -> {
            if (e.getColumn().getKey() == null) {
                getUI().ifPresent(ui -> ui.navigate("feature-views/" + e.getItem().getId()));
            }
        });

        refreshGrid();
        add(grid);
    }

    private HorizontalLayout createActions(FeatureViewDefinition view) {
        Button detail = new Button("Detail", e ->
                getUI().ifPresent(ui -> ui.navigate("feature-views/" + view.getId())));
        detail.addThemeVariants(ButtonVariant.LUMO_SMALL);

        Button delete = new Button("Delete", e -> confirmDelete(view));
        delete.addThemeVariants(ButtonVariant.LUMO_SMALL, ButtonVariant.LUMO_ERROR);

        return new HorizontalLayout(detail, delete);
    }

    private void openCreateDialog() {
        Dialog dialog = new Dialog();
        dialog.setHeaderTitle("Create Feature View");
        dialog.setWidth("600px");

        Select<EntityDefinition> entitySelect = new Select<>();
        entitySelect.setLabel("Entity");
        entitySelect.setItemLabelGenerator(EntityDefinition::getName);
        entitySelect.setItems(registryService.listEntities());
        entitySelect.setRequiredIndicatorVisible(true);

        TextField nameField = new TextField("View Name");
        nameField.setRequired(true);
        IntegerField versionField = new IntegerField("Version");
        versionField.setValue(1);
        versionField.setMin(1);
        TextField descField = new TextField("Description");
        TextField modelField = new TextField("Model Name");
        Select<String> frameworkField = new Select<>();
        frameworkField.setLabel("ML Framework");
        frameworkField.setItems("xgboost", "lightgbm", "pytorch", "tensorflow", "sklearn", "other");

        // Feature selection - updates when entity changes
        MultiSelectListBox<FeatureDefinition> featureSelect = new MultiSelectListBox<>();
        featureSelect.setItemLabelGenerator(f -> f.getName() + " (" + f.getDtype() + ")");
        featureSelect.setHeight("200px");

        entitySelect.addValueChangeListener(e -> {
            if (e.getValue() != null) {
                List<FeatureDefinition> features = registryService.listActiveFeaturesByEntity(e.getValue().getId());
                featureSelect.setItems(features);
            }
        });

        FormLayout form = new FormLayout(entitySelect, nameField, versionField, descField, modelField, frameworkField);
        dialog.add(form);
        dialog.add(new H3("Select Features (in order)"));
        dialog.add(featureSelect);

        Button save = new Button("Create", e -> {
            try {
                if (entitySelect.getValue() == null) {
                    Notification.show("Select an entity", 3000, Notification.Position.BOTTOM_START);
                    return;
                }
                Set<FeatureDefinition> selected = featureSelect.getSelectedItems();
                if (selected.isEmpty()) {
                    Notification.show("Select at least one feature", 3000, Notification.Position.BOTTOM_START);
                    return;
                }

                FeatureViewDefinition view = new FeatureViewDefinition();
                view.setName(nameField.getValue());
                view.setVersion(versionField.getValue());
                view.setEntity(entitySelect.getValue());
                view.setDescription(descField.getValue());
                view.setModelName(modelField.getValue());
                view.setMlFramework(frameworkField.getValue());

                List<UUID> featureIds = new ArrayList<>(selected.stream()
                        .map(FeatureDefinition::getId).toList());

                registryService.createFeatureView(view, featureIds);
                dialog.close();
                refreshGrid();
                Notification.show("Feature view created", 3000, Notification.Position.BOTTOM_START)
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

    private void confirmDelete(FeatureViewDefinition view) {
        ConfirmDialog dialog = new ConfirmDialog();
        dialog.setHeader("Delete Feature View");
        dialog.setText("Delete view '" + view.getName() + " v" + view.getVersion() + "'?");
        dialog.setCancelable(true);
        dialog.setConfirmText("Delete");
        dialog.setConfirmButtonTheme("error primary");
        dialog.addConfirmListener(e -> {
            try {
                registryService.deleteFeatureView(view.getId());
                refreshGrid();
                Notification.show("Feature view deleted", 3000, Notification.Position.BOTTOM_START);
            } catch (Exception ex) {
                Notification.show("Error: " + ex.getMessage(), 5000, Notification.Position.BOTTOM_START)
                        .addThemeVariants(NotificationVariant.LUMO_ERROR);
            }
        });
        dialog.open();
    }

    private void refreshGrid() {
        grid.setItems(registryService.listFeatureViews());
    }
}
