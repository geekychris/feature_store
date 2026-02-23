package com.platform.featurestore.domain;

import jakarta.persistence.*;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Entity
@Table(name = "feature_views", uniqueConstraints = {
    @UniqueConstraint(columnNames = {"name", "version"})
})
public class FeatureViewDefinition {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    @Column(name = "view_id")
    private UUID id;

    @Column(nullable = false, length = 200)
    private String name;

    @Column(nullable = false)
    private int version = 1;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "entity_id", nullable = false)
    private EntityDefinition entity;

    private String description;

    @Column(name = "model_name", length = 200)
    private String modelName;

    @Column(name = "ml_framework", length = 50)
    private String mlFramework;

    @Column(name = "vector_length")
    private Integer vectorLength;

    @Column(name = "thrift_schema", columnDefinition = "TEXT")
    private String thriftSchema;

    @Column(nullable = false, length = 20)
    private String status = "ACTIVE";

    @OneToMany(mappedBy = "featureView", cascade = CascadeType.ALL, orphanRemoval = true)
    @OrderBy("position ASC")
    private List<FeatureViewMember> members = new ArrayList<>();

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

    public int getVersion() { return version; }
    public void setVersion(int version) { this.version = version; }

    public EntityDefinition getEntity() { return entity; }
    public void setEntity(EntityDefinition entity) { this.entity = entity; }

    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }

    public String getModelName() { return modelName; }
    public void setModelName(String modelName) { this.modelName = modelName; }

    public String getMlFramework() { return mlFramework; }
    public void setMlFramework(String mlFramework) { this.mlFramework = mlFramework; }

    public Integer getVectorLength() { return vectorLength; }
    public void setVectorLength(Integer vectorLength) { this.vectorLength = vectorLength; }

    public String getThriftSchema() { return thriftSchema; }
    public void setThriftSchema(String thriftSchema) { this.thriftSchema = thriftSchema; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public List<FeatureViewMember> getMembers() { return members; }
    public void setMembers(List<FeatureViewMember> members) { this.members = members; }

    public OffsetDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(OffsetDateTime createdAt) { this.createdAt = createdAt; }

    public OffsetDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(OffsetDateTime updatedAt) { this.updatedAt = updatedAt; }
}
