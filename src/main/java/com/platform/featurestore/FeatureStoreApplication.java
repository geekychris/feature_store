package com.platform.featurestore;

import com.vaadin.flow.component.page.AppShellConfigurator;
import com.vaadin.flow.component.page.Push;
import com.vaadin.flow.theme.Theme;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@Push
@Theme("feature-store")
public class FeatureStoreApplication implements AppShellConfigurator {

    public static void main(String[] args) {
        SpringApplication.run(FeatureStoreApplication.class, args);
    }
}
