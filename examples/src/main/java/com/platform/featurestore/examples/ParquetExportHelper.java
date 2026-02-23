package com.platform.featurestore.examples;

import com.platform.featurestore.examples.MerchantFraudDataGenerator.MerchantRow;
import org.apache.avro.Schema;
import org.apache.avro.SchemaBuilder;
import org.apache.avro.generic.GenericData;
import org.apache.avro.generic.GenericRecord;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.avro.AvroParquetWriter;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Exports merchant data to Parquet in two forms:
 * <ul>
 *   <li><b>Attribute form</b> (tall/narrow): one row per (entity_id, feature_name, value)</li>
 *   <li><b>Materialized form</b> (wide): one row per entity with all features as columns</li>
 * </ul>
 */
public class ParquetExportHelper {

    /**
     * Write attribute-form Parquet: each row is (entity_id, feature_name, value_float, event_time).
     * This matches the Iceberg offline store schema.
     */
    public static void writeAttributeForm(List<MerchantRow> rows, List<String> featureNames,
                                           File outputFile) throws IOException {
        Schema schema = SchemaBuilder.record("FeatureRecord")
                .fields()
                .requiredString("entity_id")
                .requiredString("feature_name")
                .optionalDouble("value_float")
                .requiredString("event_time")
                .endRecord();

        Path path = new Path(outputFile.toURI());
        try (ParquetWriter<GenericRecord> writer = AvroParquetWriter.<GenericRecord>builder(path)
                .withSchema(schema)
                .withCompressionCodec(CompressionCodecName.SNAPPY)
                .withConf(new Configuration())
                .build()) {

            String eventTime = java.time.OffsetDateTime.now().toString();
            for (MerchantRow row : rows) {
                for (int f = 0; f < featureNames.size(); f++) {
                    GenericRecord record = new GenericData.Record(schema);
                    record.put("entity_id", row.entityId());
                    record.put("feature_name", featureNames.get(f));
                    record.put("value_float", row.features()[f]);
                    record.put("event_time", eventTime);
                    writer.write(record);
                }
            }
        }
    }

    /**
     * Write materialized-form Parquet: one row per entity with all features as named columns.
     * Schema: entity_id, feature_1, feature_2, ..., is_high_risk.
     */
    public static void writeMaterializedForm(List<MerchantRow> rows, List<String> featureNames,
                                              File outputFile) throws IOException {
        SchemaBuilder.FieldAssembler<Schema> fields = SchemaBuilder.record("MerchantFeatures")
                .fields()
                .requiredString("entity_id");
        for (String name : featureNames) {
            fields = fields.requiredDouble(name);
        }
        Schema schema = fields.requiredInt("is_high_risk").endRecord();

        Path path = new Path(outputFile.toURI());
        try (ParquetWriter<GenericRecord> writer = AvroParquetWriter.<GenericRecord>builder(path)
                .withSchema(schema)
                .withCompressionCodec(CompressionCodecName.SNAPPY)
                .withConf(new Configuration())
                .build()) {

            for (MerchantRow row : rows) {
                GenericRecord record = new GenericData.Record(schema);
                record.put("entity_id", row.entityId());
                for (int f = 0; f < featureNames.size(); f++) {
                    record.put(featureNames.get(f), row.features()[f]);
                }
                record.put("is_high_risk", row.label());
                writer.write(record);
            }
        }
    }
}
