package com.platform.featurestore.config;

import org.rocksdb.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import jakarta.annotation.PreDestroy;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Configuration
public class RocksDBConfig {

    private static final Logger log = LoggerFactory.getLogger(RocksDBConfig.class);

    public static final String CF_FEATURE_VECTORS = "feature_vectors";
    public static final String CF_FEATURE_SCALARS = "feature_scalars";
    public static final String CF_EMBEDDINGS = "embeddings";
    public static final String CF_SCHEMAS = "schemas";

    @Value("${feature-store.rocksdb.path:/tmp/feature-store/rocksdb}")
    private String dbPath;

    @Value("${feature-store.rocksdb.vector-cache-size-mb:2048}")
    private int vectorCacheSizeMb;

    @Value("${feature-store.rocksdb.scalar-cache-size-mb:1024}")
    private int scalarCacheSizeMb;

    private RocksDB db;
    private final List<ColumnFamilyHandle> cfHandles = new ArrayList<>();

    @Bean
    public RocksDB rocksDB() throws RocksDBException, IOException {
        RocksDB.loadLibrary();

        Path path = Path.of(dbPath);
        Files.createDirectories(path);

        // Column family descriptors
        List<ColumnFamilyDescriptor> cfDescriptors = new ArrayList<>();

        // Default CF (required by RocksDB)
        cfDescriptors.add(new ColumnFamilyDescriptor(
                RocksDB.DEFAULT_COLUMN_FAMILY, new ColumnFamilyOptions()));

        // Feature vectors CF - optimized for point lookups
        ColumnFamilyOptions vectorOpts = new ColumnFamilyOptions()
                .setTableFormatConfig(new BlockBasedTableConfig()
                        .setFilterPolicy(new BloomFilter(10))
                        .setBlockCache(new LRUCache(vectorCacheSizeMb * 1024L * 1024L))
                        .setBlockSize(8192))
                .setCompressionType(CompressionType.LZ4_COMPRESSION)
                .setWriteBufferSize(64 * 1024 * 1024L)
                .setMaxWriteBufferNumber(3);
        cfDescriptors.add(new ColumnFamilyDescriptor(
                CF_FEATURE_VECTORS.getBytes(), vectorOpts));

        // Feature scalars CF
        ColumnFamilyOptions scalarOpts = new ColumnFamilyOptions()
                .setTableFormatConfig(new BlockBasedTableConfig()
                        .setFilterPolicy(new BloomFilter(10))
                        .setBlockCache(new LRUCache(scalarCacheSizeMb * 1024L * 1024L)))
                .setCompressionType(CompressionType.LZ4_COMPRESSION);
        cfDescriptors.add(new ColumnFamilyDescriptor(
                CF_FEATURE_SCALARS.getBytes(), scalarOpts));

        // Embeddings CF - higher compression for large float arrays
        ColumnFamilyOptions embeddingOpts = new ColumnFamilyOptions()
                .setCompressionType(CompressionType.ZSTD_COMPRESSION)
                .setTargetFileSizeBase(128 * 1024 * 1024L);
        cfDescriptors.add(new ColumnFamilyDescriptor(
                CF_EMBEDDINGS.getBytes(), embeddingOpts));

        // Schemas CF
        cfDescriptors.add(new ColumnFamilyDescriptor(
                CF_SCHEMAS.getBytes(), new ColumnFamilyOptions()));

        DBOptions dbOptions = new DBOptions()
                .setCreateIfMissing(true)
                .setCreateMissingColumnFamilies(true)
                .setMaxOpenFiles(500)
                .setMaxBackgroundJobs(4);

        db = RocksDB.open(dbOptions, dbPath, cfDescriptors, cfHandles);
        log.info("RocksDB opened at {} with {} column families", dbPath, cfHandles.size());

        return db;
    }

    @Bean
    public Map<String, ColumnFamilyHandle> columnFamilyHandles() throws RocksDBException, IOException {
        // Ensure DB is initialized first
        rocksDB();

        Map<String, ColumnFamilyHandle> handleMap = new HashMap<>();
        String[] names = {"default", CF_FEATURE_VECTORS, CF_FEATURE_SCALARS, CF_EMBEDDINGS, CF_SCHEMAS};
        for (int i = 0; i < cfHandles.size(); i++) {
            handleMap.put(names[i], cfHandles.get(i));
        }
        return handleMap;
    }

    @PreDestroy
    public void close() {
        log.info("Closing RocksDB...");
        cfHandles.forEach(ColumnFamilyHandle::close);
        if (db != null) {
            db.close();
        }
    }
}
