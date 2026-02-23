package com.platform.featurestore.ui;

import com.vaadin.flow.component.applayout.AppLayout;
import com.vaadin.flow.component.applayout.DrawerToggle;
import com.vaadin.flow.component.html.H1;
import com.vaadin.flow.component.html.Nav;
import com.vaadin.flow.component.html.Span;
import com.vaadin.flow.component.icon.VaadinIcon;
import com.vaadin.flow.component.orderedlayout.FlexComponent;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.sidenav.SideNav;
import com.vaadin.flow.component.sidenav.SideNavItem;
import com.vaadin.flow.router.Layout;

@Layout
public class MainLayout extends AppLayout {

    public MainLayout() {
        createHeader();
        createDrawer();
    }

    private void createHeader() {
        H1 logo = new H1("Feature Store");
        logo.getStyle()
                .set("font-size", "var(--lumo-font-size-l)")
                .set("margin", "0");

        Span subtitle = new Span("Admin Console");
        subtitle.getStyle()
                .set("font-size", "var(--lumo-font-size-s)")
                .set("color", "var(--lumo-secondary-text-color)");

        VerticalLayout titleBlock = new VerticalLayout(logo, subtitle);
        titleBlock.setSpacing(false);
        titleBlock.setPadding(false);

        HorizontalLayout header = new HorizontalLayout(new DrawerToggle(), titleBlock);
        header.setDefaultVerticalComponentAlignment(FlexComponent.Alignment.CENTER);
        header.setWidthFull();
        header.setPadding(true);
        header.getStyle().set("padding-left", "var(--lumo-space-m)");

        addToNavbar(header);
    }

    private void createDrawer() {
        SideNav nav = new SideNav();

        nav.addItem(new SideNavItem("Dashboard", DashboardView.class, VaadinIcon.DASHBOARD.create()));
        nav.addItem(new SideNavItem("Entities", EntityListView.class, VaadinIcon.DATABASE.create()));
        nav.addItem(new SideNavItem("Features", FeatureListView.class, VaadinIcon.LIST.create()));
        nav.addItem(new SideNavItem("Feature Views", FeatureViewListView.class, VaadinIcon.GRID_BIG.create()));
        nav.addItem(new SideNavItem("Value Lookup", FeatureLookupView.class, VaadinIcon.SEARCH.create()));
        nav.addItem(new SideNavItem("Statistics", StatisticsView.class, VaadinIcon.CHART.create()));
        nav.addItem(new SideNavItem("Training", TrainingView.class, VaadinIcon.COGS.create()));

        addToDrawer(new VerticalLayout(nav));
    }
}
