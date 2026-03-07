mod config;
mod dataset;
mod engine;
mod ffi;
mod grpc_service;
mod metrics;

pub mod proto {
    tonic::include_proto!("scorched");
}

use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use tonic::transport::Server;
use tracing_subscriber::EnvFilter;

use crate::config::Config;
use crate::dataset::DatasetManager;
use crate::engine::create_scorer;
use crate::grpc_service::{DatasetServiceImpl, ScoreServiceImpl, ServiceState, StatsServiceImpl};
use crate::metrics::Metrics;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse();

    // ---- Logging ----
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.log_level));

    if config.json_log {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }

    // ---- Model info ----
    let model_info = ffi::ModelInfo::query();
    tracing::info!(
        trees = model_info.num_trees,
        features = model_info.num_features,
        user_features = model_info.num_user_features,
        item_features = model_info.num_item_features,
        base_score = model_info.base_score,
        "Model loaded via FFI"
    );

    // ---- Initialize components ----
    let metrics = Arc::new(Metrics::new());
    let scorer = create_scorer(config.cuda);
    let datasets = Arc::new(DatasetManager::new(
        config.max_datasets,
        model_info.num_item_features,
    ));

    if config.cuda {
        metrics.cuda_available.set(1);
    }

    let state = Arc::new(ServiceState {
        scorer,
        datasets,
        metrics: metrics.clone(),
        start_time: Instant::now(),
    });

    // ---- Metrics server (HTTP) ----
    let metrics_port = config.metrics_port;
    let metrics_handle = tokio::spawn(metrics::serve_metrics(metrics.clone(), metrics_port));

    // ---- gRPC server ----
    let addr = format!("0.0.0.0:{}", config.port).parse()?;

    let score_svc = proto::score_service_server::ScoreServiceServer::new(ScoreServiceImpl {
        state: state.clone(),
    })
    .max_decoding_message_size(config.max_message_size);

    let dataset_svc =
        proto::dataset_service_server::DatasetServiceServer::new(DatasetServiceImpl {
            state: state.clone(),
        })
        .max_decoding_message_size(config.max_message_size);

    let stats_svc = proto::stats_service_server::StatsServiceServer::new(StatsServiceImpl {
        state: state.clone(),
    });

    tracing::info!(
        port = config.port,
        metrics_port = config.metrics_port,
        backend = config.backend_name(),
        max_datasets = config.max_datasets,
        workers = config.effective_worker_threads(),
        "scorched runtime starting"
    );

    Server::builder()
        .add_service(score_svc)
        .add_service(dataset_svc)
        .add_service(stats_svc)
        .serve_with_shutdown(addr, async {
            tokio::signal::ctrl_c()
                .await
                .expect("failed to install CTRL+C handler");
            tracing::info!("Shutdown signal received, draining...");
        })
        .await?;

    metrics_handle.abort();
    tracing::info!("scorched runtime stopped");
    Ok(())
}
