package com.platform.featurestore.config;

import org.apache.hadoop.conf.Configuration;
import org.apache.iceberg.catalog.Catalog;
import org.apache.iceberg.hadoop.HadoopCatalog;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;

import jakarta.annotation.PreDestroy;
import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

@org.springframework.context.annotation.Configuration
public class IcebergConfig {

    private static final Logger log = LoggerFactory.getLogger(IcebergConfig.class);

    @Value("${feature-store.iceberg.warehouse:/tmp/feature-store/iceberg}")
    private String warehouse;

    @Value("${feature-store.iceberg.catalog-name:feature_store}")
    private String catalogName;

    private HadoopCatalog catalog;

    @org.springframework.context.annotation.Bean
    public Catalog icebergCatalog() throws IOException {
        Path warehousePath = Path.of(warehouse);
        Files.createDirectories(warehousePath);

        Configuration hadoopConf = new Configuration();
        catalog = new HadoopCatalog();
        catalog.setConf(hadoopConf);
        catalog.initialize(catalogName, com.google.common.collect.ImmutableMap.of(
                "warehouse", warehousePath.toUri().toString()
        ));

        log.info("Iceberg HadoopCatalog initialized at {}", warehouse);
        return catalog;
    }

    @PreDestroy
    public void close() throws IOException {
        if (catalog != null) {
            catalog.close();
        }
    }
}
