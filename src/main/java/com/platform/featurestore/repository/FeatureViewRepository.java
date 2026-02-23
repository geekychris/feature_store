package com.platform.featurestore.repository;

import com.platform.featurestore.domain.FeatureViewDefinition;
import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Repository
public interface FeatureViewRepository extends JpaRepository<FeatureViewDefinition, UUID> {

    @Override
    @EntityGraph(attributePaths = {"entity", "members", "members.feature"})
    List<FeatureViewDefinition> findAll();

    @Override
    @EntityGraph(attributePaths = {"entity", "members", "members.feature"})
    Optional<FeatureViewDefinition> findById(UUID id);

    @EntityGraph(attributePaths = {"entity", "members", "members.feature"})
    Optional<FeatureViewDefinition> findByNameAndVersion(String name, int version);

    @EntityGraph(attributePaths = {"entity"})
    List<FeatureViewDefinition> findByName(String name);

    @EntityGraph(attributePaths = {"entity", "members", "members.feature"})
    Optional<FeatureViewDefinition> findFirstByNameOrderByVersionDesc(String name);

    @EntityGraph(attributePaths = {"entity", "members", "members.feature"})
    List<FeatureViewDefinition> findByEntityId(UUID entityId);

    @EntityGraph(attributePaths = {"entity"})
    List<FeatureViewDefinition> findByStatus(String status);
}
