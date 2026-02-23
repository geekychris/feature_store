package com.platform.featurestore.store.offline;

import org.apache.iceberg.*;
import org.apache.iceberg.catalog.Catalog;
import org.apache.iceberg.catalog.Namespace;
import org.apache.iceberg.catalog.TableIdentifier;
import org.apache.iceberg.data.GenericRecord;
import org.apache.iceberg.data.IcebergGenerics;
import org.apache.iceberg.data.Record;
import org.apache.iceberg.data.parquet.GenericParquetWriter;
import org.apache.iceberg.expressions.Expressions;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.io.DataWriter;
import org.apache.iceberg.io.OutputFile;
import org.apache.iceberg.parquet.Parquet;
import org.apache.iceberg.types.Types;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Offline store backed by Apache Iceberg.
 * Provides point-in-time correct training data generation.
 */
@Component
public class IcebergOfflineStore {

    private static final Logger log = LoggerFactory.getLogger(IcebergOfflineStore.class);
    private static final String TABLE_NAME = "feature_records";
    private static final Namespace NAMESPACE = Namespace.of("features");

    private final Catalog catalog;

    // Iceberg schema for feature records
    private static final Schema FEATURE_RECORD_SCHEMA = new Schema(
            Types.NestedField.required(1, "entity_type", Types.StringType.get()),
            Types.NestedField.required(2, "entity_id", Types.StringType.get()),
            Types.NestedField.required(3, "feature_name", Types.StringType.get()),
            Types.NestedField.optional(4, "value_float", Types.DoubleType.get()),
            Types.NestedField.optional(5, "value_string", Types.StringType.get()),
            Types.NestedField.required(6, "event_time", Types.TimestampType.withZone()),
            Types.NestedField.required(7, "created_at", Types.TimestampType.withZone()),
            Types.NestedField.optional(8, "pipeline_id", Types.StringType.get()),
            Types.NestedField.optional(9, "view_version", Types.IntegerType.get())
    );

    private static final PartitionSpec PARTITION_SPEC = PartitionSpec.builderFor(FEATURE_RECORD_SCHEMA)
            .identity("entity_type")
            .day("event_time")
            .build();

    public IcebergOfflineStore(Catalog catalog) {
        this.catalog = catalog;
    }

    /**
     * Ensure the feature records table exists, creating it if necessary.
     */
    public Table ensureTable() {
        TableIdentifier tableId = TableIdentifier.of(NAMESPACE, TABLE_NAME);
        if (catalog.tableExists(tableId)) {
            return catalog.loadTable(tableId);
        }
        log.info("Creating Iceberg table: {}", tableId);
        return catalog.createTable(tableId, FEATURE_RECORD_SCHEMA, PARTITION_SPEC);
    }

    /**
     * Write feature records to the offline store.
     */
    public void writeFeatureRecords(List<FeatureRecordData> records) throws IOException {
        Table table = ensureTable();
        String dataFile = table.location() + "/data/" + UUID.randomUUID() + ".parquet";
        OutputFile outputFile = table.io().newOutputFile(dataFile);

        DataWriter<Record> writer = Parquet.writeData(outputFile)
                .schema(FEATURE_RECORD_SCHEMA)
                .createWriterFunc(GenericParquetWriter::buildWriter)
                .overwrite()
                .withSpec(PartitionSpec.unpartitioned())
                .build();

        try {
            for (FeatureRecordData record : records) {
                GenericRecord row = GenericRecord.create(FEATURE_RECORD_SCHEMA);
                row.setField("entity_type", record.entityType());
                row.setField("entity_id", record.entityId());
                row.setField("feature_name", record.featureName());
                row.setField("value_float", record.valueFloat());
                row.setField("value_string", record.valueString());
                row.setField("event_time", record.eventTime());
                row.setField("created_at", record.createdAt());
                row.setField("pipeline_id", record.pipelineId());
                row.setField("view_version", record.viewVersion());
                writer.write(row);
            }
        } finally {
            writer.close();
        }

        // Commit the file to the table
        table.newAppend()
                .appendFile(writer.toDataFile())
                .commit();

        log.info("Wrote {} feature records to Iceberg", records.size());
    }

    /**
     * Generate point-in-time correct training data.
     * <p>
     * For each label event, finds the most recent value of each feature
     * that existed BEFORE the label event time. This prevents data leakage.
     */
    public List<Map<String, Object>> generateTrainingDataset(
            String entityType,
            List<String> featureNames,
            List<LabelEvent> labelEvents) throws IOException {

        Table table = ensureTable();

        // Find time bounds
        OffsetDateTime minTime = labelEvents.stream()
                .map(LabelEvent::eventTime)
                .min(OffsetDateTime::compareTo)
                .orElseThrow();
        OffsetDateTime maxTime = labelEvents.stream()
                .map(LabelEvent::eventTime)
                .max(OffsetDateTime::compareTo)
                .orElseThrow();

        // Scan feature history within the time window
        CloseableIterable<Record> scan = IcebergGenerics.read(table)
                .where(Expressions.and(
                        Expressions.equal("entity_type", entityType),
                        Expressions.greaterThanOrEqual("event_time",
                                minTime.minusDays(180).toInstant().toEpochMilli() * 1000),
                        Expressions.lessThanOrEqual("event_time",
                                maxTime.toInstant().toEpochMilli() * 1000)
                ))
                .build();

        // Build in-memory feature history index: entityId -> featureName -> sorted list of (time, value)
        Map<String, Map<String, TreeMap<OffsetDateTime, Double>>> history = new HashMap<>();

        try {
            for (Record record : scan) {
                String eid = (String) record.getField("entity_id");
                String fname = (String) record.getField("feature_name");
                Double value = (Double) record.getField("value_float");
                Object eventTimeObj = record.getField("event_time");

                if (value == null || !featureNames.contains(fname)) continue;

                OffsetDateTime eventTime = toOffsetDateTime(eventTimeObj);
                history.computeIfAbsent(eid, k -> new HashMap<>())
                        .computeIfAbsent(fname, k -> new TreeMap<>())
                        .put(eventTime, value);
            }
        } finally {
            scan.close();
        }

        // ASOF join: for each label event, find latest feature value before it
        List<Map<String, Object>> rows = new ArrayList<>();

        for (LabelEvent label : labelEvents) {
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("entity_id", label.entityId());
            row.put("event_time", label.eventTime());
            row.put("label", label.label());

            Map<String, TreeMap<OffsetDateTime, Double>> entityHistory =
                    history.getOrDefault(label.entityId(), Map.of());

            for (String fname : featureNames) {
                TreeMap<OffsetDateTime, Double> featureHistory =
                        entityHistory.getOrDefault(fname, new TreeMap<>());

                // Floor entry: latest value at or before the label time
                Map.Entry<OffsetDateTime, Double> entry = featureHistory.floorEntry(label.eventTime());
                row.put(fname, entry != null ? entry.getValue() : Double.NaN);
            }

            rows.add(row);
        }

        return rows;
    }

    private OffsetDateTime toOffsetDateTime(Object obj) {
        if (obj instanceof OffsetDateTime odt) return odt;
        if (obj instanceof Long micros) {
            return Instant.ofEpochMilli(micros / 1000).atOffset(ZoneOffset.UTC);
        }
        throw new IllegalArgumentException("Unexpected time type: " + obj.getClass());
    }

    // Data transfer records

    public record FeatureRecordData(
            String entityType,
            String entityId,
            String featureName,
            Double valueFloat,
            String valueString,
            OffsetDateTime eventTime,
            OffsetDateTime createdAt,
            String pipelineId,
            Integer viewVersion
    ) {}

    public record LabelEvent(
            String entityId,
            OffsetDateTime eventTime,
            int label
    ) {}
}
