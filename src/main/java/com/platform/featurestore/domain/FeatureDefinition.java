package com.platform.featurestore.domain;

import jakarta.persistence.*;
import java.time.OffsetDateTime;
import java.util.UUID;

@Entity
@Table(name = "features", uniqueConstraints = {
    @UniqueConstraint(columnNames = {"name", "entity_id"})
})
public class FeatureDefinition {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    @Column(name = "feature_id")
    private UUID id;

    @Column(nullable = false, length = 200)
    private String name;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "entity_id", nullable = false)
    private EntityDefinition entity;

    @Column(nullable = false, length = 30)
    private String dtype;

    private String description;

    @Column(length = 100)
    private String owner;

    @Column(name = "source_pipeline", length = 200)
    private String sourcePipeline;

    @Column(name = "source_query", columnDefinition = "TEXT")
    private String sourceQuery;

    @Column(name = "update_frequency", length = 50)
    private String updateFrequency;

    @Column(name = "max_age_seconds")
    private Integer maxAgeSeconds;

    @Column(name = "default_value", columnDefinition = "TEXT")
    private String defaultValue;

    @Column(nullable = false, columnDefinition = "TEXT DEFAULT '{}'")
    private String tags = "{}";

    @Column(nullable = false, length = 20)
    private String status = "ACTIVE";

    @Column(name = "deprecated_at")
    private OffsetDateTime deprecatedAt;

    @Column(name = "deprecation_message")
    private String deprecationMessage;

    @Column(nullable = false)
    private int version = 1;

    @Column(name = "created_at", nullable = false)
    private OffsetDateTime createdAt;

    @Column(name = "updated_at", nullable = false)
    private OffsetDateTime updatedAt;

    @PrePersist
    protected void onCreate() {
        createdAt = OffsetDateTime.now();
        updatedAt = OffsetDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = OffsetDateTime.now();
    }

    // Getters and setters

    public UUID getId() { return id; }
    public void setId(UUID id) { this.id = id; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public EntityDefinition getEntity() { return entity; }
    public void setEntity(EntityDefinition entity) { this.entity = entity; }

    public String getDtype() { return dtype; }
    public void setDtype(String dtype) { this.dtype = dtype; }

    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }

    public String getOwner() { return owner; }
    public void setOwner(String owner) { this.owner = owner; }

    public String getSourcePipeline() { return sourcePipeline; }
    public void setSourcePipeline(String sourcePipeline) { this.sourcePipeline = sourcePipeline; }

    public String getSourceQuery() { return sourceQuery; }
    public void setSourceQuery(String sourceQuery) { this.sourceQuery = sourceQuery; }

    public String getUpdateFrequency() { return updateFrequency; }
    public void setUpdateFrequency(String updateFrequency) { this.updateFrequency = updateFrequency; }

    public Integer getMaxAgeSeconds() { return maxAgeSeconds; }
    public void setMaxAgeSeconds(Integer maxAgeSeconds) { this.maxAgeSeconds = maxAgeSeconds; }

    public String getDefaultValue() { return defaultValue; }
    public void setDefaultValue(String defaultValue) { this.defaultValue = defaultValue; }

    public String getTags() { return tags; }
    public void setTags(String tags) { this.tags = tags; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public OffsetDateTime getDeprecatedAt() { return deprecatedAt; }
    public void setDeprecatedAt(OffsetDateTime deprecatedAt) { this.deprecatedAt = deprecatedAt; }

    public String getDeprecationMessage() { return deprecationMessage; }
    public void setDeprecationMessage(String deprecationMessage) { this.deprecationMessage = deprecationMessage; }

    public int getVersion() { return version; }
    public void setVersion(int version) { this.version = version; }

    public OffsetDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(OffsetDateTime createdAt) { this.createdAt = createdAt; }

    public OffsetDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(OffsetDateTime updatedAt) { this.updatedAt = updatedAt; }
}
