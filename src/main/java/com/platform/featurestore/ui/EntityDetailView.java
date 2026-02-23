package com.platform.featurestore.ui;

import com.platform.featurestore.domain.*;
import com.platform.featurestore.service.FeatureRegistryService;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.html.H3;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.tabs.Tab;
import com.vaadin.flow.component.tabs.Tabs;
import com.vaadin.flow.router.*;

import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.UUID;

@Route(value = "entities/:entityId", layout = MainLayout.class)
@PageTitle("Entity Detail | Feature Store")
public class EntityDetailView extends VerticalLayout implements BeforeEnterObserver {

    private static final DateTimeFormatter FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    private final FeatureRegistryService registryService;

    private final H3 title = new H3();
    private final Span metadata = new Span();
    private final VerticalLayout tabContent = new VerticalLayout();

    public EntityDetailView(FeatureRegistryService registryService) {
        this.registryService = registryService;
        setPadding(true);
        setSpacing(true);

        add(title, metadata);

        Tabs tabs = new Tabs();
        Tab featuresTab = new Tab("Features");
        Tab viewsTab = new Tab("Feature Views");
        Tab eventsTab = new Tab("Lifecycle Events");
        tabs.add(featuresTab, viewsTab, eventsTab);

        tabContent.setPadding(false);
        add(tabs, tabContent);

        tabs.addSelectedChangeListener(e -> {
            // Will be populated in beforeEnter
        });
    }

    @Override
    public void beforeEnter(BeforeEnterEvent event) {
        String entityIdStr = event.getRouteParameters().get("entityId").orElse("");
        try {
            UUID entityId = UUID.fromString(entityIdStr);
            registryService.getEntity(entityId).ifPresentOrElse(
                    entity -> populateView(entity),
                    () -> {
                        title.setText("Entity not found");
                        tabContent.removeAll();
                    }
            );
        } catch (IllegalArgumentException e) {
            title.setText("Invalid entity ID");
        }
    }

    private void populateView(EntityDefinition entity) {
        title.setText("Entity: " + entity.getName());
        metadata.setText("Join Key: " + entity.getJoinKey() + " (" + entity.getJoinKeyType() + ")"
                + (entity.getDescription() != null ? " — " + entity.getDescription() : ""));

        tabContent.removeAll();

        // Features grid
        List<FeatureDefinition> features = registryService.listFeaturesByEntity(entity.getId());
        Grid<FeatureDefinition> featGrid = new Grid<>();
        featGrid.addColumn(FeatureDefinition::getName).setHeader("Name").setSortable(true);
        featGrid.addColumn(FeatureDefinition::getDtype).setHeader("Type");
        featGrid.addColumn(FeatureDefinition::getStatus).setHeader("Status");
        featGrid.addColumn(FeatureDefinition::getOwner).setHeader("Owner");
        featGrid.addColumn(f -> f.getVersion()).setHeader("Version");
        featGrid.setItems(features);
        featGrid.setHeight("300px");
        featGrid.setWidthFull();

        // Views grid
        List<FeatureViewDefinition> views = registryService.listFeatureViewsByEntity(entity.getId());
        Grid<FeatureViewDefinition> viewGrid = new Grid<>();
        viewGrid.addColumn(FeatureViewDefinition::getName).setHeader("Name").setSortable(true);
        viewGrid.addColumn(FeatureViewDefinition::getVersion).setHeader("Version");
        viewGrid.addColumn(v -> v.getVectorLength() != null ? v.getVectorLength() : 0).setHeader("Vector Length");
        viewGrid.addColumn(FeatureViewDefinition::getStatus).setHeader("Status");
        viewGrid.setItems(views);
        viewGrid.setHeight("250px");
        viewGrid.setWidthFull();

        // Events grid
        List<FeatureLifecycleEvent> events = registryService.getEventsForEntity(entity.getId());
        Grid<FeatureLifecycleEvent> eventGrid = new Grid<>();
        eventGrid.addColumn(e -> e.getOccurredAt() != null ? e.getOccurredAt().format(FMT) : "")
                .setHeader("Time");
        eventGrid.addColumn(FeatureLifecycleEvent::getEventType).setHeader("Event");
        eventGrid.addColumn(FeatureLifecycleEvent::getActor).setHeader("Actor");
        eventGrid.setItems(events);
        eventGrid.setHeight("250px");
        eventGrid.setWidthFull();

        tabContent.add(
                new H3("Features (" + features.size() + ")"), featGrid,
                new H3("Feature Views (" + views.size() + ")"), viewGrid,
                new H3("Lifecycle Events (" + events.size() + ")"), eventGrid
        );
    }
}
