-- ============================================================
-- FEATURE STORE METADATA REGISTRY DDL
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Entity types (merchant, consumer, device, transaction)
CREATE TABLE entities (
    entity_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            VARCHAR(100) NOT NULL UNIQUE,
    description     TEXT,
    join_key        VARCHAR(100) NOT NULL,
    join_key_type   VARCHAR(20)  NOT NULL,
    tags            TEXT         NOT NULL DEFAULT '{}',
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Individual feature definitions
CREATE TABLE features (
    feature_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name                VARCHAR(200) NOT NULL,
    entity_id           UUID NOT NULL REFERENCES entities(entity_id),
    dtype               VARCHAR(30)  NOT NULL,
    description         TEXT,
    owner               VARCHAR(100),
    source_pipeline     VARCHAR(200),
    source_query        TEXT,
    update_frequency    VARCHAR(50),
    max_age_seconds     INTEGER,
    default_value       TEXT,
    tags                TEXT NOT NULL DEFAULT '{}',
    status              VARCHAR(20) NOT NULL DEFAULT 'ACTIVE'
                            CHECK (status IN ('DRAFT','ACTIVE','DEPRECATED','ARCHIVED')),
    deprecated_at       TIMESTAMP WITH TIME ZONE,
    deprecation_message TEXT,
    version             INTEGER NOT NULL DEFAULT 1,
    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE (name, entity_id)
);

-- Feature views: named groups of features used by a model
CREATE TABLE feature_views (
    view_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            VARCHAR(200) NOT NULL,
    version         INTEGER      NOT NULL DEFAULT 1,
    entity_id       UUID         NOT NULL REFERENCES entities(entity_id),
    description     TEXT,
    model_name      VARCHAR(200),
    ml_framework    VARCHAR(50),
    vector_length   INTEGER,
    thrift_schema   TEXT,
    status          VARCHAR(20)  NOT NULL DEFAULT 'ACTIVE',
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE (name, version)
);

-- Ordered feature membership in a view
CREATE TABLE feature_view_members (
    view_id          UUID     NOT NULL REFERENCES feature_views(view_id),
    feature_id       UUID     NOT NULL REFERENCES features(feature_id),
    position         SMALLINT NOT NULL,
    alias            VARCHAR(100),
    transform        VARCHAR(200),
    transform_params TEXT,
    is_required      BOOLEAN  NOT NULL DEFAULT TRUE,
    PRIMARY KEY (view_id, feature_id),
    UNIQUE (view_id, position)
);

-- Lifecycle events for audit
CREATE TABLE feature_lifecycle_events (
    event_id        UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type     VARCHAR(30) NOT NULL,
    entity_ref_id   UUID        NOT NULL,
    event_type      VARCHAR(50) NOT NULL,
    actor           VARCHAR(100),
    details         TEXT        NOT NULL DEFAULT '{}',
    occurred_at     TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Statistics tracked per feature
CREATE TABLE feature_statistics (
    stat_id         UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    feature_id      UUID        NOT NULL REFERENCES features(feature_id),
    computed_at     TIMESTAMP WITH TIME ZONE NOT NULL,
    window_start    TIMESTAMP WITH TIME ZONE,
    window_end      TIMESTAMP WITH TIME ZONE,
    count           BIGINT,
    null_count      BIGINT,
    mean            DOUBLE PRECISION,
    stddev          DOUBLE PRECISION,
    min_value       DOUBLE PRECISION,
    max_value       DOUBLE PRECISION,
    p25             DOUBLE PRECISION,
    p50             DOUBLE PRECISION,
    p75             DOUBLE PRECISION,
    p95             DOUBLE PRECISION,
    p99             DOUBLE PRECISION,
    histogram_bins  TEXT,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_feature_stats_feature ON feature_statistics(feature_id, computed_at DESC);
CREATE INDEX idx_features_entity ON features(entity_id, status);
CREATE INDEX idx_fvm_view ON feature_view_members(view_id, position);
CREATE INDEX idx_lifecycle_ref ON feature_lifecycle_events(entity_ref_id, occurred_at DESC);
