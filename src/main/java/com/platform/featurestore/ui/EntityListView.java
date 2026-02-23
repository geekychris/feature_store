package com.platform.featurestore.ui;

import com.platform.featurestore.domain.EntityDefinition;
import com.platform.featurestore.service.FeatureRegistryService;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.button.ButtonVariant;
import com.vaadin.flow.component.confirmdialog.ConfirmDialog;
import com.vaadin.flow.component.dialog.Dialog;
import com.vaadin.flow.component.formlayout.FormLayout;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.html.H3;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.notification.NotificationVariant;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.select.Select;
import com.vaadin.flow.component.textfield.TextField;
import com.vaadin.flow.router.PageTitle;
import com.vaadin.flow.router.Route;

import java.time.format.DateTimeFormatter;

@Route(value = "entities", layout = MainLayout.class)
@PageTitle("Entities | Feature Store")
public class EntityListView extends VerticalLayout {

    private static final DateTimeFormatter FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm");
    private final FeatureRegistryService registryService;
    private final Grid<EntityDefinition> grid = new Grid<>();

    public EntityListView(FeatureRegistryService registryService) {
        this.registryService = registryService;
        setPadding(true);
        setSpacing(true);

        // Header
        Button createBtn = new Button("Create Entity", e -> openCreateDialog());
        createBtn.addThemeVariants(ButtonVariant.LUMO_PRIMARY);

        HorizontalLayout header = new HorizontalLayout(new H3("Entities"), createBtn);
        header.setDefaultVerticalComponentAlignment(Alignment.BASELINE);
        add(header);

        // Grid
        grid.addColumn(EntityDefinition::getName).setHeader("Name").setSortable(true).setFlexGrow(1);
        grid.addColumn(EntityDefinition::getJoinKey).setHeader("Join Key").setFlexGrow(0).setWidth("150px");
        grid.addColumn(EntityDefinition::getJoinKeyType).setHeader("Key Type").setFlexGrow(0).setWidth("100px");
        grid.addColumn(e -> e.getDescription() != null ? e.getDescription() : "—")
                .setHeader("Description").setFlexGrow(2);
        grid.addColumn(e -> e.getCreatedAt() != null ? e.getCreatedAt().format(FMT) : "")
                .setHeader("Created").setFlexGrow(0).setWidth("150px");
        grid.addComponentColumn(this::createActions).setHeader("Actions").setFlexGrow(0).setWidth("200px");

        grid.setWidthFull();
        grid.setHeight("600px");

        // Click-through to detail
        grid.addItemClickListener(e -> {
            if (e.getColumn().getKey() == null) { // not the actions column
                getUI().ifPresent(ui -> ui.navigate("entities/" + e.getItem().getId()));
            }
        });

        refreshGrid();
        add(grid);
    }

    private HorizontalLayout createActions(EntityDefinition entity) {
        Button edit = new Button("Edit", e -> openEditDialog(entity));
        edit.addThemeVariants(ButtonVariant.LUMO_SMALL);

        Button delete = new Button("Delete", e -> confirmDelete(entity));
        delete.addThemeVariants(ButtonVariant.LUMO_SMALL, ButtonVariant.LUMO_ERROR);

        return new HorizontalLayout(edit, delete);
    }

    private void openCreateDialog() {
        Dialog dialog = new Dialog();
        dialog.setHeaderTitle("Create Entity");

        TextField nameField = new TextField("Name");
        nameField.setRequired(true);
        TextField descField = new TextField("Description");
        TextField joinKeyField = new TextField("Join Key");
        joinKeyField.setRequired(true);
        Select<String> joinKeyTypeField = new Select<>();
        joinKeyTypeField.setLabel("Key Type");
        joinKeyTypeField.setItems("STRING", "INT64", "UUID");
        joinKeyTypeField.setValue("STRING");

        FormLayout form = new FormLayout(nameField, descField, joinKeyField, joinKeyTypeField);
        dialog.add(form);

        Button save = new Button("Create", e -> {
            try {
                EntityDefinition entity = new EntityDefinition();
                entity.setName(nameField.getValue());
                entity.setDescription(descField.getValue());
                entity.setJoinKey(joinKeyField.getValue());
                entity.setJoinKeyType(joinKeyTypeField.getValue());
                registryService.createEntity(entity);
                dialog.close();
                refreshGrid();
                Notification.show("Entity created", 3000, Notification.Position.BOTTOM_START)
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

    private void openEditDialog(EntityDefinition entity) {
        Dialog dialog = new Dialog();
        dialog.setHeaderTitle("Edit Entity");

        TextField nameField = new TextField("Name");
        nameField.setValue(entity.getName());
        TextField descField = new TextField("Description");
        descField.setValue(entity.getDescription() != null ? entity.getDescription() : "");
        TextField joinKeyField = new TextField("Join Key");
        joinKeyField.setValue(entity.getJoinKey());
        Select<String> joinKeyTypeField = new Select<>();
        joinKeyTypeField.setLabel("Key Type");
        joinKeyTypeField.setItems("STRING", "INT64", "UUID");
        joinKeyTypeField.setValue(entity.getJoinKeyType());

        FormLayout form = new FormLayout(nameField, descField, joinKeyField, joinKeyTypeField);
        dialog.add(form);

        Button save = new Button("Save", e -> {
            try {
                registryService.updateEntity(entity.getId(),
                        nameField.getValue(), descField.getValue(),
                        joinKeyField.getValue(), joinKeyTypeField.getValue());
                dialog.close();
                refreshGrid();
                Notification.show("Entity updated", 3000, Notification.Position.BOTTOM_START)
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

    private void confirmDelete(EntityDefinition entity) {
        ConfirmDialog dialog = new ConfirmDialog();
        dialog.setHeader("Delete Entity");
        dialog.setText("Delete entity '" + entity.getName() + "'? This cannot be undone.");
        dialog.setCancelable(true);
        dialog.setConfirmText("Delete");
        dialog.setConfirmButtonTheme("error primary");
        dialog.addConfirmListener(e -> {
            try {
                registryService.deleteEntity(entity.getId());
                refreshGrid();
                Notification.show("Entity deleted", 3000, Notification.Position.BOTTOM_START);
            } catch (Exception ex) {
                Notification.show("Error: " + ex.getMessage(), 5000, Notification.Position.BOTTOM_START)
                        .addThemeVariants(NotificationVariant.LUMO_ERROR);
            }
        });
        dialog.open();
    }

    private void refreshGrid() {
        grid.setItems(registryService.listEntities());
    }
}
