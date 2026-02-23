package com.platform.featurestore.repository;

import com.platform.featurestore.domain.FeatureStatistics;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Repository
public interface FeatureStatisticsRepository extends JpaRepository<FeatureStatistics, UUID> {
    Optional<FeatureStatistics> findFirstByFeatureIdOrderByComputedAtDesc(UUID featureId);
    List<FeatureStatistics> findByFeatureIdOrderByComputedAtDesc(UUID featureId);
}
