package com.platform.featurestore.repository;

import com.platform.featurestore.domain.FeatureDefinition;
import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Repository
public interface FeatureDefinitionRepository extends JpaRepository<FeatureDefinition, UUID> {

    @Override
    @EntityGraph(attributePaths = {"entity"})
    List<FeatureDefinition> findAll();

    @Override
    @EntityGraph(attributePaths = {"entity"})
    Optional<FeatureDefinition> findById(UUID id);

    @EntityGraph(attributePaths = {"entity"})
    List<FeatureDefinition> findByEntityIdAndStatus(UUID entityId, String status);

    @EntityGraph(attributePaths = {"entity"})
    Optional<FeatureDefinition> findByNameAndEntityId(String name, UUID entityId);

    @EntityGraph(attributePaths = {"entity"})
    List<FeatureDefinition> findByEntityId(UUID entityId);

    @EntityGraph(attributePaths = {"entity"})
    List<FeatureDefinition> findByOwner(String owner);

    @EntityGraph(attributePaths = {"entity"})
    List<FeatureDefinition> findByStatus(String status);
}
