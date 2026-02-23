package com.platform.featurestore.config;

import org.apache.catalina.connector.Connector;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.embedded.tomcat.TomcatServletWebServerFactory;
import org.springframework.boot.web.server.WebServerFactoryCustomizer;
import org.springframework.context.annotation.Configuration;

/**
 * Adds a second Tomcat connector so the admin UI is accessible on a dedicated port.
 * The REST API remains on the primary server.port (8085).
 * The admin UI is also reachable at http://localhost:{admin.port}/ui/
 */
@Configuration
public class AdminPortConfig implements WebServerFactoryCustomizer<TomcatServletWebServerFactory> {

    @Value("${admin.port:8086}")
    private int adminPort;

    @Override
    public void customize(TomcatServletWebServerFactory factory) {
        Connector connector = new Connector(TomcatServletWebServerFactory.DEFAULT_PROTOCOL);
        connector.setPort(adminPort);
        factory.addAdditionalTomcatConnectors(connector);
    }
}
