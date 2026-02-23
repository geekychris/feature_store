package com.platform.featurestore.domain;

import jakarta.persistence.*;
import java.time.OffsetDateTime;
import java.util.UUID;

@Entity
@Table(name = "feature_statistics")
public class FeatureStatistics {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    @Column(name = "stat_id")
    private UUID id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "feature_id", nullable = false)
    private FeatureDefinition feature;

    @Column(name = "computed_at", nullable = false)
    private OffsetDateTime computedAt;

    @Column(name = "window_start")
    private OffsetDateTime windowStart;

    @Column(name = "window_end")
    private OffsetDateTime windowEnd;

    private Long count;

    @Column(name = "null_count")
    private Long nullCount;

    private Double mean;
    private Double stddev;

    @Column(name = "min_value")
    private Double minValue;

    @Column(name = "max_value")
    private Double maxValue;

    private Double p25;
    private Double p50;
    private Double p75;
    private Double p95;
    private Double p99;

    @Column(name = "histogram_bins", columnDefinition = "TEXT")
    private String histogramBins;

    @Column(name = "created_at", nullable = false)
    private OffsetDateTime createdAt;

    @PrePersist
    protected void onCreate() {
        createdAt = OffsetDateTime.now();
    }

    // Getters and setters

    public UUID getId() { return id; }
    public void setId(UUID id) { this.id = id; }

    public FeatureDefinition getFeature() { return feature; }
    public void setFeature(FeatureDefinition feature) { this.feature = feature; }

    public OffsetDateTime getComputedAt() { return computedAt; }
    public void setComputedAt(OffsetDateTime computedAt) { this.computedAt = computedAt; }

    public OffsetDateTime getWindowStart() { return windowStart; }
    public void setWindowStart(OffsetDateTime windowStart) { this.windowStart = windowStart; }

    public OffsetDateTime getWindowEnd() { return windowEnd; }
    public void setWindowEnd(OffsetDateTime windowEnd) { this.windowEnd = windowEnd; }

    public Long getCount() { return count; }
    public void setCount(Long count) { this.count = count; }

    public Long getNullCount() { return nullCount; }
    public void setNullCount(Long nullCount) { this.nullCount = nullCount; }

    public Double getMean() { return mean; }
    public void setMean(Double mean) { this.mean = mean; }

    public Double getStddev() { return stddev; }
    public void setStddev(Double stddev) { this.stddev = stddev; }

    public Double getMinValue() { return minValue; }
    public void setMinValue(Double minValue) { this.minValue = minValue; }

    public Double getMaxValue() { return maxValue; }
    public void setMaxValue(Double maxValue) { this.maxValue = maxValue; }

    public Double getP25() { return p25; }
    public void setP25(Double p25) { this.p25 = p25; }

    public Double getP50() { return p50; }
    public void setP50(Double p50) { this.p50 = p50; }

    public Double getP75() { return p75; }
    public void setP75(Double p75) { this.p75 = p75; }

    public Double getP95() { return p95; }
    public void setP95(Double p95) { this.p95 = p95; }

    public Double getP99() { return p99; }
    public void setP99(Double p99) { this.p99 = p99; }

    public String getHistogramBins() { return histogramBins; }
    public void setHistogramBins(String histogramBins) { this.histogramBins = histogramBins; }

    public OffsetDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(OffsetDateTime createdAt) { this.createdAt = createdAt; }
}
