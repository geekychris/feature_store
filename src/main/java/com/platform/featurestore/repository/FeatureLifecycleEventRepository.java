package com.platform.featurestore.repository;

import com.platform.featurestore.domain.FeatureLifecycleEvent;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.UUID;

@Repository
public interface FeatureLifecycleEventRepository extends JpaRepository<FeatureLifecycleEvent, UUID> {
    List<FeatureLifecycleEvent> findByEntityRefIdOrderByOccurredAtDesc(UUID entityRefId);
    List<FeatureLifecycleEvent> findByEntityTypeAndEventType(String entityType, String eventType);
}
