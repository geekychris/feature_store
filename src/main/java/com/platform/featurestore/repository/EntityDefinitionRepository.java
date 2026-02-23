package com.platform.featurestore.repository;

import com.platform.featurestore.domain.EntityDefinition;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;
import java.util.UUID;

@Repository
public interface EntityDefinitionRepository extends JpaRepository<EntityDefinition, UUID> {
    Optional<EntityDefinition> findByName(String name);
    boolean existsByName(String name);
}
