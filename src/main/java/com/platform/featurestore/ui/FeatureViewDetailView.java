package com.platform.featurestore.ui;

import com.platform.featurestore.domain.FeatureViewDefinition;
import com.platform.featurestore.domain.FeatureViewMember;
import com.platform.featurestore.service.FeatureRegistryService;
import com.vaadin.flow.component.grid.Grid;
import com.vaadin.flow.component.html.H3;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.router.*;

import java.util.Comparator;
import java.util.UUID;

@Route(value = "feature-views/:viewId", layout = MainLayout.class)
@PageTitle("Feature View Detail | Feature Store")
public class FeatureViewDetailView extends VerticalLayout implements BeforeEnterObserver {

    private final FeatureRegistryService registryService;
    private final H3 title = new H3();
    private final VerticalLayout content = new VerticalLayout();

    public FeatureViewDetailView(FeatureRegistryService registryService) {
        this.registryService = registryService;
        setPadding(true);
        setSpacing(true);
        add(title, content);
        content.setPadding(false);
    }

    @Override
    public void beforeEnter(BeforeEnterEvent event) {
        String viewIdStr = event.getRouteParameters().get("viewId").orElse("");
        try {
            UUID viewId = UUID.fromString(viewIdStr);
            registryService.getFeatureView(viewId).ifPresentOrElse(
                    this::populateView,
                    () -> title.setText("Feature view not found")
            );
        } catch (IllegalArgumentException e) {
            title.setText("Invalid view ID");
        }
    }

    private void populateView(FeatureViewDefinition view) {
        title.setText("Feature View: " + view.getName() + " v" + view.getVersion());
        content.removeAll();

        // Metadata
        VerticalLayout meta = new VerticalLayout();
        meta.setSpacing(false);
        meta.setPadding(false);
        meta.add(createMetaRow("Entity", view.getEntity() != null ? view.getEntity().getName() : "—"));
        meta.add(createMetaRow("Description", view.getDescription() != null ? view.getDescription() : "—"));
        meta.add(createMetaRow("Model", view.getModelName() != null ? view.getModelName() : "—"));
        meta.add(createMetaRow("ML Framework", view.getMlFramework() != null ? view.getMlFramework() : "—"));
        meta.add(createMetaRow("Vector Length", String.valueOf(view.getVectorLength() != null ? view.getVectorLength() : 0)));
        meta.add(createMetaRow("Status", view.getStatus()));

        // Compute schema hash
        try {
            var schema = registryService.buildViewSchema(view);
            meta.add(createMetaRow("Schema Hash", String.valueOf(schema.getSchemaHash())));
        } catch (Exception e) {
            // ignore
        }

        content.add(meta);

        // Members grid
        content.add(new H3("Feature Members"));
        Grid<FeatureViewMember> memberGrid = new Grid<>();
        memberGrid.addColumn(FeatureViewMember::getPosition).setHeader("Position").setFlexGrow(0).setWidth("80px");
        memberGrid.addColumn(m -> m.getFeature() != null ? m.getFeature().getName() : "—")
                .setHeader("Feature Name").setFlexGrow(1);
        memberGrid.addColumn(m -> m.getFeature() != null ? m.getFeature().getDtype() : "—")
                .setHeader("Type").setFlexGrow(0).setWidth("100px");
        memberGrid.addColumn(m -> m.getAlias() != null ? m.getAlias() : "—")
                .setHeader("Alias").setFlexGrow(0).setWidth("150px");
        memberGrid.addColumn(m -> m.getTransform() != null ? m.getTransform() : "—")
                .setHeader("Transform").setFlexGrow(0).setWidth("150px");
        memberGrid.addColumn(m -> m.isRequired() ? "Yes" : "No")
                .setHeader("Required").setFlexGrow(0).setWidth("80px");

        var members = view.getMembers().stream()
                .sorted(Comparator.comparingInt(FeatureViewMember::getPosition))
                .toList();
        memberGrid.setItems(members);
        memberGrid.setWidthFull();
        memberGrid.setAllRowsVisible(true);

        content.add(memberGrid);
    }

    private HorizontalLayout createMetaRow(String label, String value) {
        Span labelSpan = new Span(label + ": ");
        labelSpan.getStyle().set("font-weight", "bold").set("min-width", "140px");
        Span valueSpan = new Span(value);
        HorizontalLayout row = new HorizontalLayout(labelSpan, valueSpan);
        row.setSpacing(true);
        return row;
    }
}
