package com.platform.featurestore.ui;

import com.platform.featurestore.domain.FeatureLifecycleEvent;
import com.platform.featurestore.service.FeatureRegistryService;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.html.Div;
import com.vaadin.flow.component.html.H2;
import com.vaadin.flow.component.html.H3;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.router.PageTitle;
import com.vaadin.flow.router.Route;

import java.time.format.DateTimeFormatter;

@Route(value = "", layout = MainLayout.class)
@PageTitle("Dashboard | Feature Store")
public class DashboardView extends VerticalLayout {

    private static final DateTimeFormatter FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    public DashboardView(FeatureRegistryService registryService) {
        setPadding(true);
        setSpacing(true);

        // Stats cards
        long entityCount = registryService.listEntities().size();
        long featureCount = registryService.listAllFeatures().size();
        long activeFeatures = registryService.listAllFeatures().stream()
                .filter(f -> "ACTIVE".equals(f.getStatus())).count();
        long deprecatedFeatures = featureCount - activeFeatures;
        long viewCount = registryService.listFeatureViews().size();

        HorizontalLayout cards = new HorizontalLayout();
        cards.setWidthFull();
        cards.add(
                createCard(String.valueOf(entityCount), "Entities"),
                createCard(String.valueOf(featureCount), "Features",
                        activeFeatures + " active, " + deprecatedFeatures + " deprecated"),
                createCard(String.valueOf(viewCount), "Feature Views"),
                createCard("—", "Training Runs")
        );
        add(cards);

        // Recent events
        add(new H3("Recent Activity"));
        Grid<FeatureLifecycleEvent> eventsGrid = new Grid<>();
        eventsGrid.addColumn(e -> e.getOccurredAt() != null ? e.getOccurredAt().format(FMT) : "")
                .setHeader("Time").setFlexGrow(0).setWidth("180px");
        eventsGrid.addColumn(FeatureLifecycleEvent::getEntityType).setHeader("Type").setFlexGrow(0).setWidth("120px");
        eventsGrid.addColumn(FeatureLifecycleEvent::getEventType).setHeader("Event").setFlexGrow(0).setWidth("120px");
        eventsGrid.addColumn(e -> e.getEntityRefId() != null ? e.getEntityRefId().toString().substring(0, 8) + "…" : "")
                .setHeader("Ref ID").setFlexGrow(0).setWidth("120px");
        eventsGrid.addColumn(FeatureLifecycleEvent::getActor).setHeader("Actor").setFlexGrow(1);

        eventsGrid.setItems(registryService.getRecentEvents(25));
        eventsGrid.setHeight("400px");
        add(eventsGrid);
    }

    private Div createCard(String value, String label) {
        return createCard(value, label, null);
    }

    private Div createCard(String value, String label, String subtitle) {
        H2 val = new H2(value);
        Span lbl = new Span(label);
        Div card = new Div(val, lbl);
        if (subtitle != null) {
            Span sub = new Span(subtitle);
            sub.getStyle().set("font-size", "var(--lumo-font-size-xs)")
                    .set("color", "var(--lumo-tertiary-text-color)");
            card.add(sub);
        }
        card.addClassName("dashboard-card");
        card.getStyle().set("flex", "1");
        return card;
    }
}
