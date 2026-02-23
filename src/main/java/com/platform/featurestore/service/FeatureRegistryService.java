package com.platform.featurestore.service;

import com.platform.featurestore.domain.*;
import com.platform.featurestore.proto.ViewSchema;
import com.platform.featurestore.repository.*;
import com.platform.featurestore.store.online.RocksDBFeatureStore;
import org.rocksdb.RocksDBException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class FeatureRegistryService {

    private static final Logger log = LoggerFactory.getLogger(FeatureRegistryService.class);

    private final EntityDefinitionRepository entityRepo;
    private final FeatureDefinitionRepository featureRepo;
    private final FeatureViewRepository viewRepo;
    private final FeatureLifecycleEventRepository lifecycleRepo;
    private final FeatureStatisticsRepository statsRepo;
    private final RocksDBFeatureStore rocksStore;

    public FeatureRegistryService(EntityDefinitionRepository entityRepo,
                                   FeatureDefinitionRepository featureRepo,
                                   FeatureViewRepository viewRepo,
                                   FeatureLifecycleEventRepository lifecycleRepo,
                                   FeatureStatisticsRepository statsRepo,
                                   RocksDBFeatureStore rocksStore) {
        this.entityRepo = entityRepo;
        this.featureRepo = featureRepo;
        this.viewRepo = viewRepo;
        this.lifecycleRepo = lifecycleRepo;
        this.statsRepo = statsRepo;
        this.rocksStore = rocksStore;
    }

    // --- Entity operations ---

    @Transactional
    public EntityDefinition createEntity(EntityDefinition entity) {
        EntityDefinition saved = entityRepo.save(entity);
        recordEvent("ENTITY", saved.getId(), "CREATED", "system");
        return saved;
    }

    public Optional<EntityDefinition> getEntity(UUID id) {
        return entityRepo.findById(id);
    }

    public Optional<EntityDefinition> getEntityByName(String name) {
        return entityRepo.findByName(name);
    }

    public List<EntityDefinition> listEntities() {
        return entityRepo.findAll();
    }

    @Transactional
    public EntityDefinition updateEntity(UUID id, String name, String description,
                                          String joinKey, String joinKeyType) {
        EntityDefinition entity = entityRepo.findById(id)
                .orElseThrow(() -> new NoSuchElementException("Entity not found: " + id));
        if (name != null) entity.setName(name);
        if (description != null) entity.setDescription(description);
        if (joinKey != null) entity.setJoinKey(joinKey);
        if (joinKeyType != null) entity.setJoinKeyType(joinKeyType);
        EntityDefinition saved = entityRepo.save(entity);
        recordEvent("ENTITY", id, "UPDATED", "admin_ui");
        return saved;
    }

    @Transactional
    public void deleteEntity(UUID id) {
        List<FeatureDefinition> features = featureRepo.findByEntityId(id);
        if (!features.isEmpty()) {
            throw new IllegalStateException(
                    "Cannot delete entity with " + features.size() + " features. Remove features first.");
        }
        entityRepo.deleteById(id);
        recordEvent("ENTITY", id, "DELETED", "admin_ui");
    }

    // --- Feature operations ---

    @Transactional
    public FeatureDefinition createFeature(FeatureDefinition feature) {
        FeatureDefinition saved = featureRepo.save(feature);
        recordEvent("FEATURE", saved.getId(), "CREATED", feature.getOwner());
        return saved;
    }

    public Optional<FeatureDefinition> getFeature(UUID id) {
        return featureRepo.findById(id);
    }

    public List<FeatureDefinition> listFeaturesByEntity(UUID entityId) {
        return featureRepo.findByEntityId(entityId);
    }

    public List<FeatureDefinition> listActiveFeaturesByEntity(UUID entityId) {
        return featureRepo.findByEntityIdAndStatus(entityId, "ACTIVE");
    }

    public List<FeatureDefinition> listAllFeatures() {
        return featureRepo.findAll();
    }

    @Transactional
    public FeatureDefinition updateFeature(UUID id, String description, String owner,
                                            String updateFrequency, Integer maxAgeSeconds,
                                            String defaultValue) {
        FeatureDefinition feature = featureRepo.findById(id)
                .orElseThrow(() -> new NoSuchElementException("Feature not found: " + id));
        if (description != null) feature.setDescription(description);
        if (owner != null) feature.setOwner(owner);
        if (updateFrequency != null) feature.setUpdateFrequency(updateFrequency);
        if (maxAgeSeconds != null) feature.setMaxAgeSeconds(maxAgeSeconds);
        if (defaultValue != null) feature.setDefaultValue(defaultValue);
        FeatureDefinition saved = featureRepo.save(feature);
        recordEvent("FEATURE", id, "UPDATED", feature.getOwner());
        return saved;
    }

    @Transactional
    public FeatureDefinition deprecateFeature(UUID featureId, String message) {
        FeatureDefinition feature = featureRepo.findById(featureId)
                .orElseThrow(() -> new NoSuchElementException("Feature not found: " + featureId));
        feature.setStatus("DEPRECATED");
        feature.setDeprecatedAt(java.time.OffsetDateTime.now());
        feature.setDeprecationMessage(message);
        FeatureDefinition saved = featureRepo.save(feature);
        recordEvent("FEATURE", featureId, "DEPRECATED", feature.getOwner());
        return saved;
    }

    // --- Feature View operations ---

    @Transactional
    public FeatureViewDefinition createFeatureView(FeatureViewDefinition view,
                                                     List<UUID> featureIds) {
        // Set vector length
        view.setVectorLength(featureIds.size());
        FeatureViewDefinition saved = viewRepo.save(view);

        // Add ordered members
        for (int i = 0; i < featureIds.size(); i++) {
            FeatureDefinition feature = featureRepo.findById(featureIds.get(i))
                    .orElseThrow(() -> new NoSuchElementException("Feature not found"));
            FeatureViewMember member = new FeatureViewMember();
            member.setFeatureView(saved);
            member.setFeature(feature);
            member.setPosition((short) i);
            member.setRequired(true);
            saved.getMembers().add(member);
        }

        saved = viewRepo.save(saved);

        // Compute and store schema hash
        ViewSchema schema = buildViewSchema(saved);
        try {
            rocksStore.putSchema(schema);
        } catch (RocksDBException e) {
            log.error("Failed to store schema in RocksDB", e);
        }

        recordEvent("FEATURE_VIEW", saved.getId(), "CREATED", "system");
        return saved;
    }

    public Optional<FeatureViewDefinition> getFeatureView(UUID id) {
        return viewRepo.findById(id);
    }

    public Optional<FeatureViewDefinition> getFeatureViewByNameAndVersion(String name, int version) {
        return viewRepo.findByNameAndVersion(name, version);
    }

    public Optional<FeatureViewDefinition> getLatestFeatureView(String name) {
        return viewRepo.findFirstByNameOrderByVersionDesc(name);
    }

    public List<FeatureViewDefinition> listFeatureViews() {
        return viewRepo.findAll();
    }

    public List<FeatureViewDefinition> listFeatureViewsByEntity(UUID entityId) {
        return viewRepo.findByEntityId(entityId);
    }

    @Transactional
    public void deleteFeatureView(UUID id) {
        viewRepo.deleteById(id);
        recordEvent("FEATURE_VIEW", id, "DELETED", "admin_ui");
    }

    // --- Schema operations ---

    /**
     * Build a ViewSchema protobuf from a FeatureViewDefinition.
     * The schema hash is MD5(sorted feature names) truncated to 8 hex chars → int32.
     */
    public ViewSchema buildViewSchema(FeatureViewDefinition view) {
        List<String> featureNames = view.getMembers().stream()
                .sorted(Comparator.comparingInt(FeatureViewMember::getPosition))
                .map(m -> m.getAlias() != null ? m.getAlias() : m.getFeature().getName())
                .collect(Collectors.toList());

        List<String> featureDtypes = view.getMembers().stream()
                .sorted(Comparator.comparingInt(FeatureViewMember::getPosition))
                .map(m -> m.getFeature().getDtype())
                .collect(Collectors.toList());

        int hash = computeSchemaHash(featureNames);

        return ViewSchema.newBuilder()
                .setViewName(view.getName())
                .setVersion(view.getVersion())
                .addAllFeatureNames(featureNames)
                .addAllFeatureDtypes(featureDtypes)
                .setSchemaHash(hash)
                .setCreatedAtMs(System.currentTimeMillis())
                .build();
    }

    /**
     * Compute schema hash from ordered feature names.
     * MD5 of comma-joined sorted names, truncated to 8 hex chars, converted to int.
     */
    public static int computeSchemaHash(List<String> featureNames) {
        try {
            String key = String.join(",", featureNames);
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] digest = md.digest(key.getBytes(StandardCharsets.UTF_8));
            String hexStr = bytesToHex(digest).substring(0, 8);
            return (int) (Long.parseLong(hexStr, 16) % Integer.MAX_VALUE);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("MD5 not available", e);
        }
    }

    private static String bytesToHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }

    // --- Lifecycle events ---

    public void recordEvent(String entityType, UUID entityRefId, String eventType, String actor) {
        FeatureLifecycleEvent event = new FeatureLifecycleEvent();
        event.setEntityType(entityType);
        event.setEntityRefId(entityRefId);
        event.setEventType(eventType);
        event.setActor(actor);
        lifecycleRepo.save(event);
    }

    public List<FeatureLifecycleEvent> getEventsForEntity(UUID entityRefId) {
        return lifecycleRepo.findByEntityRefIdOrderByOccurredAtDesc(entityRefId);
    }

    public List<FeatureLifecycleEvent> getRecentEvents(int limit) {
        return lifecycleRepo.findAll(
                org.springframework.data.domain.PageRequest.of(0, limit,
                        org.springframework.data.domain.Sort.by(
                                org.springframework.data.domain.Sort.Direction.DESC, "occurredAt")))
                .getContent();
    }

    // --- Statistics ---

    public Optional<FeatureStatistics> getLatestStatistics(UUID featureId) {
        return statsRepo.findFirstByFeatureIdOrderByComputedAtDesc(featureId);
    }

    public List<FeatureStatistics> getStatisticsHistory(UUID featureId) {
        return statsRepo.findByFeatureIdOrderByComputedAtDesc(featureId);
    }
}
