package com.platform.featurestore.controller;

import com.platform.featurestore.domain.*;
import com.platform.featurestore.dto.Dtos.*;
import com.platform.featurestore.service.FeatureRegistryService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/v1")
public class RegistryController {

    private final FeatureRegistryService registryService;

    public RegistryController(FeatureRegistryService registryService) {
        this.registryService = registryService;
    }

    // --- Entities ---

    @PostMapping("/entities")
    public ResponseEntity<EntityResponse> createEntity(@RequestBody CreateEntityRequest req) {
        EntityDefinition entity = new EntityDefinition();
        entity.setName(req.name());
        entity.setDescription(req.description());
        entity.setJoinKey(req.joinKey());
        entity.setJoinKeyType(req.joinKeyType());
        EntityDefinition saved = registryService.createEntity(entity);
        return ResponseEntity.status(HttpStatus.CREATED).body(toResponse(saved));
    }

    @GetMapping("/entities")
    public List<EntityResponse> listEntities() {
        return registryService.listEntities().stream().map(this::toResponse).collect(Collectors.toList());
    }

    @GetMapping("/entities/{id}")
    public ResponseEntity<EntityResponse> getEntity(@PathVariable UUID id) {
        return registryService.getEntity(id)
                .map(e -> ResponseEntity.ok(toResponse(e)))
                .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/entities/by-name/{name}")
    public ResponseEntity<EntityResponse> getEntityByName(@PathVariable String name) {
        return registryService.getEntityByName(name)
                .map(e -> ResponseEntity.ok(toResponse(e)))
                .orElse(ResponseEntity.notFound().build());
    }

    // --- Features ---

    @PostMapping("/features")
    public ResponseEntity<FeatureResponse> createFeature(@RequestBody CreateFeatureRequest req) {
        EntityDefinition entity = registryService.getEntity(req.entityId())
                .orElseThrow(() -> new IllegalArgumentException("Entity not found: " + req.entityId()));

        FeatureDefinition feature = new FeatureDefinition();
        feature.setName(req.name());
        feature.setEntity(entity);
        feature.setDtype(req.dtype());
        feature.setDescription(req.description());
        feature.setOwner(req.owner());
        feature.setSourcePipeline(req.sourcePipeline());
        feature.setUpdateFrequency(req.updateFrequency());
        feature.setMaxAgeSeconds(req.maxAgeSeconds());
        feature.setDefaultValue(req.defaultValue());

        FeatureDefinition saved = registryService.createFeature(feature);
        return ResponseEntity.status(HttpStatus.CREATED).body(toResponse(saved));
    }

    @GetMapping("/features")
    public List<FeatureResponse> listFeatures(@RequestParam(required = false) UUID entityId) {
        if (entityId != null) {
            return registryService.listFeaturesByEntity(entityId).stream()
                    .map(this::toResponse).collect(Collectors.toList());
        }
        return List.of(); // Require entityId filter
    }

    @GetMapping("/features/{id}")
    public ResponseEntity<FeatureResponse> getFeature(@PathVariable UUID id) {
        return registryService.getFeature(id)
                .map(f -> ResponseEntity.ok(toResponse(f)))
                .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping("/features/{id}/deprecate")
    public ResponseEntity<FeatureResponse> deprecateFeature(@PathVariable UUID id,
                                                             @RequestParam String message) {
        FeatureDefinition deprecated = registryService.deprecateFeature(id, message);
        return ResponseEntity.ok(toResponse(deprecated));
    }

    // --- Feature Views ---

    @PostMapping("/feature-views")
    public ResponseEntity<FeatureViewResponse> createFeatureView(@RequestBody CreateFeatureViewRequest req) {
        EntityDefinition entity = registryService.getEntity(req.entityId())
                .orElseThrow(() -> new IllegalArgumentException("Entity not found: " + req.entityId()));

        FeatureViewDefinition view = new FeatureViewDefinition();
        view.setName(req.name());
        view.setVersion(req.version());
        view.setEntity(entity);
        view.setDescription(req.description());
        view.setModelName(req.modelName());
        view.setMlFramework(req.mlFramework());

        FeatureViewDefinition saved = registryService.createFeatureView(view, req.featureIds());
        return ResponseEntity.status(HttpStatus.CREATED).body(toViewResponse(saved));
    }

    @GetMapping("/feature-views")
    public List<FeatureViewResponse> listFeatureViews() {
        return registryService.listFeatureViews().stream()
                .map(this::toViewResponse).collect(Collectors.toList());
    }

    @GetMapping("/feature-views/{name}/{version}")
    public ResponseEntity<FeatureViewResponse> getFeatureView(@PathVariable String name,
                                                                @PathVariable int version) {
        return registryService.getFeatureViewByNameAndVersion(name, version)
                .map(v -> ResponseEntity.ok(toViewResponse(v)))
                .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/feature-views/{name}/latest")
    public ResponseEntity<FeatureViewResponse> getLatestFeatureView(@PathVariable String name) {
        return registryService.getLatestFeatureView(name)
                .map(v -> ResponseEntity.ok(toViewResponse(v)))
                .orElse(ResponseEntity.notFound().build());
    }

    // --- Mappers ---

    private EntityResponse toResponse(EntityDefinition e) {
        return new EntityResponse(e.getId(), e.getName(), e.getDescription(),
                e.getJoinKey(), e.getJoinKeyType());
    }

    private FeatureResponse toResponse(FeatureDefinition f) {
        return new FeatureResponse(f.getId(), f.getName(),
                f.getEntity().getId(), f.getDtype(), f.getDescription(),
                f.getOwner(), f.getUpdateFrequency(), f.getMaxAgeSeconds(),
                f.getStatus(), f.getVersion());
    }

    private FeatureViewResponse toViewResponse(FeatureViewDefinition v) {
        var schema = registryService.buildViewSchema(v);
        return new FeatureViewResponse(v.getId(), v.getName(), v.getVersion(),
                v.getEntity().getId(), v.getDescription(), v.getModelName(),
                v.getVectorLength() != null ? v.getVectorLength() : 0,
                schema.getSchemaHash(), v.getStatus(),
                schema.getFeatureNamesList());
    }
}
