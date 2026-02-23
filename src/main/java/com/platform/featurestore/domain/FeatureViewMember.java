package com.platform.featurestore.domain;

import jakarta.persistence.*;
import java.io.Serializable;
import java.util.Objects;
import java.util.UUID;

@Entity
@Table(name = "feature_view_members")
@IdClass(FeatureViewMember.FeatureViewMemberId.class)
public class FeatureViewMember {

    @Id
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "view_id", nullable = false)
    private FeatureViewDefinition featureView;

    @Id
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "feature_id", nullable = false)
    private FeatureDefinition feature;

    @Column(nullable = false)
    private short position;

    @Column(length = 100)
    private String alias;

    @Column(length = 200)
    private String transform;

    @Column(name = "transform_params", columnDefinition = "TEXT")
    private String transformParams;

    @Column(name = "is_required", nullable = false)
    private boolean required = true;

    // Getters and setters

    public FeatureViewDefinition getFeatureView() { return featureView; }
    public void setFeatureView(FeatureViewDefinition featureView) { this.featureView = featureView; }

    public FeatureDefinition getFeature() { return feature; }
    public void setFeature(FeatureDefinition feature) { this.feature = feature; }

    public short getPosition() { return position; }
    public void setPosition(short position) { this.position = position; }

    public String getAlias() { return alias; }
    public void setAlias(String alias) { this.alias = alias; }

    public String getTransform() { return transform; }
    public void setTransform(String transform) { this.transform = transform; }

    public String getTransformParams() { return transformParams; }
    public void setTransformParams(String transformParams) { this.transformParams = transformParams; }

    public boolean isRequired() { return required; }
    public void setRequired(boolean required) { this.required = required; }

    // Composite key class
    public static class FeatureViewMemberId implements Serializable {
        private UUID featureView;
        private UUID feature;

        public FeatureViewMemberId() {}

        public FeatureViewMemberId(UUID featureView, UUID feature) {
            this.featureView = featureView;
            this.feature = feature;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof FeatureViewMemberId that)) return false;
            return Objects.equals(featureView, that.featureView) &&
                   Objects.equals(feature, that.feature);
        }

        @Override
        public int hashCode() {
            return Objects.hash(featureView, feature);
        }
    }
}
