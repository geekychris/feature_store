use std::sync::Arc;
use std::time::Instant;

use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use crate::dataset::DatasetManager;
use crate::engine::Scorer;
use crate::metrics::{Metrics, ScoringTimer};
use crate::proto;

/// Shared state for all gRPC services.
pub struct ServiceState {
    pub scorer: Arc<dyn Scorer>,
    pub datasets: Arc<DatasetManager>,
    pub metrics: Arc<Metrics>,
    pub start_time: Instant,
}

// ============================================================
// ScoreService
// ============================================================

pub struct ScoreServiceImpl {
    pub state: Arc<ServiceState>,
}

#[tonic::async_trait]
impl proto::score_service_server::ScoreService for ScoreServiceImpl {
    async fn score_top_k(
        &self,
        request: Request<proto::ScoreRequest>,
    ) -> Result<Response<proto::ScoreResponse>, Status> {
        let req = request.into_inner();
        let dataset_name = &req.dataset_name;

        let dataset =
            self.state.datasets.get(dataset_name).ok_or_else(|| {
                Status::not_found(format!("dataset '{}' not found", dataset_name))
            })?;

        let _timer = ScoringTimer::new(
            self.state.metrics.clone(),
            dataset_name,
            "score_topk",
            self.state.scorer.backend_name(),
        );

        let k = req.top_k as usize;
        if k == 0 {
            return Err(Status::invalid_argument("top_k must be > 0"));
        }

        dataset.increment_requests();

        let results = self
            .state
            .scorer
            .score_topk(&req.user_vector, &dataset, k)
            .map_err(|e| Status::internal(e.to_string()))?;

        // Update metrics
        self.state
            .metrics
            .score_requests
            .with_label_values(&[dataset_name, "ok"])
            .inc();
        self.state
            .metrics
            .items_scored
            .with_label_values(&[dataset_name])
            .inc_by(dataset.num_items as u64);

        let latency_us = _timer.elapsed_us();

        Ok(Response::new(proto::ScoreResponse {
            candidates: results
                .iter()
                .map(|r| proto::ScoredCandidate {
                    index: r.index,
                    score: r.score,
                })
                .collect(),
            items_scored: dataset.num_items as i32,
            latency_us,
            backend: self.state.scorer.backend_name().to_string(),
            request_id: req.request_id,
        }))
    }

    async fn score_batch(
        &self,
        request: Request<proto::ScoreBatchRequest>,
    ) -> Result<Response<proto::ScoreBatchResponse>, Status> {
        let req = request.into_inner();
        let batch_start = Instant::now();

        let dataset = self.state.datasets.get(&req.dataset_name).ok_or_else(|| {
            Status::not_found(format!("dataset '{}' not found", req.dataset_name))
        })?;

        let k = req.top_k as usize;
        let mut results = Vec::with_capacity(req.users.len());

        for user_req in &req.users {
            let start = Instant::now();
            dataset.increment_requests();

            let scored = self
                .state
                .scorer
                .score_topk(&user_req.user_vector, &dataset, k)
                .map_err(|e| Status::internal(e.to_string()))?;

            results.push(proto::UserScoreResponse {
                user_id: user_req.user_id.clone(),
                candidates: scored
                    .iter()
                    .map(|r| proto::ScoredCandidate {
                        index: r.index,
                        score: r.score,
                    })
                    .collect(),
                latency_us: start.elapsed().as_micros() as i64,
            });
        }

        self.state
            .metrics
            .score_requests
            .with_label_values(&[&req.dataset_name, "ok"])
            .inc_by(req.users.len() as u64);

        Ok(Response::new(proto::ScoreBatchResponse {
            results,
            total_latency_us: batch_start.elapsed().as_micros() as i64,
            backend: self.state.scorer.backend_name().to_string(),
            request_id: req.request_id,
        }))
    }

    type ScoreAllStream =
        tokio_stream::wrappers::ReceiverStream<Result<proto::ScoreAllChunk, Status>>;

    async fn score_all(
        &self,
        request: Request<proto::ScoreRequest>,
    ) -> Result<Response<Self::ScoreAllStream>, Status> {
        let req = request.into_inner();

        let dataset = self.state.datasets.get(&req.dataset_name).ok_or_else(|| {
            Status::not_found(format!("dataset '{}' not found", req.dataset_name))
        })?;

        dataset.increment_requests();

        let scores = self
            .state
            .scorer
            .score_all(&req.user_vector, &dataset)
            .map_err(|e| Status::internal(e.to_string()))?;

        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let chunk_size = 10_000;

        tokio::spawn(async move {
            let total = scores.len();
            for (chunk_idx, chunk) in scores.chunks(chunk_size).enumerate() {
                let start_idx = chunk_idx * chunk_size;
                let is_last = start_idx + chunk.len() >= total;

                let candidates: Vec<proto::ScoredCandidate> = chunk
                    .iter()
                    .enumerate()
                    .map(|(i, &score)| proto::ScoredCandidate {
                        index: (start_idx + i) as i32,
                        score,
                    })
                    .collect();

                if tx
                    .send(Ok(proto::ScoreAllChunk {
                        candidates,
                        is_last,
                    }))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        });

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(
            rx,
        )))
    }
}

// ============================================================
// DatasetService
// ============================================================

pub struct DatasetServiceImpl {
    pub state: Arc<ServiceState>,
}

#[tonic::async_trait]
impl proto::dataset_service_server::DatasetService for DatasetServiceImpl {
    async fn load_dataset(
        &self,
        request: Request<proto::LoadDatasetRequest>,
    ) -> Result<Response<proto::LoadDatasetResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        let ds = self
            .state
            .datasets
            .load(
                req.name.clone(),
                req.candidate_matrix,
                req.num_items as usize,
                req.num_item_features as usize,
                req.replace_if_exists,
            )
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        self.update_dataset_metrics(&ds.name, ds.num_items, ds.memory_bytes);

        Ok(Response::new(proto::LoadDatasetResponse {
            success: true,
            message: format!("Loaded {} items", ds.num_items),
            dataset_name: ds.name.clone(),
            num_items: ds.num_items as i32,
            memory_bytes: ds.memory_bytes as i64,
            load_duration_us: start.elapsed().as_micros() as i64,
        }))
    }

    async fn load_dataset_from_file(
        &self,
        request: Request<proto::LoadDatasetFromFileRequest>,
    ) -> Result<Response<proto::LoadDatasetResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        let ds = self
            .state
            .datasets
            .load_from_file(
                req.name.clone(),
                &req.file_path,
                req.num_item_features as usize,
                req.replace_if_exists,
            )
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        self.update_dataset_metrics(&ds.name, ds.num_items, ds.memory_bytes);

        Ok(Response::new(proto::LoadDatasetResponse {
            success: true,
            message: format!("Loaded {} items from {}", ds.num_items, req.file_path),
            dataset_name: ds.name.clone(),
            num_items: ds.num_items as i32,
            memory_bytes: ds.memory_bytes as i64,
            load_duration_us: start.elapsed().as_micros() as i64,
        }))
    }

    async fn stream_load_dataset(
        &self,
        request: Request<Streaming<proto::DatasetChunk>>,
    ) -> Result<Response<proto::LoadDatasetResponse>, Status> {
        let start = Instant::now();
        let mut stream = request.into_inner();

        let mut name = String::new();
        let mut num_item_features = 0usize;
        let mut replace = false;
        let mut data = Vec::new();
        let mut first = true;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            if first {
                name = chunk.name;
                num_item_features = chunk.num_item_features as usize;
                replace = chunk.replace_if_exists;
                first = false;
            }
            data.extend_from_slice(&chunk.data);
        }

        if name.is_empty() {
            return Err(Status::invalid_argument("no chunks received"));
        }

        let num_items = data.len() / num_item_features;
        let ds = self
            .state
            .datasets
            .load(name.clone(), data, num_items, num_item_features, replace)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        self.update_dataset_metrics(&ds.name, ds.num_items, ds.memory_bytes);

        Ok(Response::new(proto::LoadDatasetResponse {
            success: true,
            message: format!("Loaded {} items via stream", ds.num_items),
            dataset_name: ds.name.clone(),
            num_items: ds.num_items as i32,
            memory_bytes: ds.memory_bytes as i64,
            load_duration_us: start.elapsed().as_micros() as i64,
        }))
    }

    async fn unload_dataset(
        &self,
        request: Request<proto::UnloadDatasetRequest>,
    ) -> Result<Response<proto::UnloadDatasetResponse>, Status> {
        let name = &request.into_inner().name;
        self.state
            .datasets
            .unload(name)
            .map_err(|e| Status::not_found(e.to_string()))?;

        // Clean up metrics
        let _ = self
            .state
            .metrics
            .dataset_items
            .remove_label_values(&[name]);
        let _ = self
            .state
            .metrics
            .dataset_memory_bytes
            .remove_label_values(&[name]);
        self.state
            .metrics
            .loaded_datasets
            .set(self.state.datasets.count() as i64);

        Ok(Response::new(proto::UnloadDatasetResponse {
            success: true,
            message: format!("Unloaded dataset '{}'", name),
        }))
    }

    async fn list_datasets(
        &self,
        _request: Request<proto::ListDatasetsRequest>,
    ) -> Result<Response<proto::ListDatasetsResponse>, Status> {
        let datasets = self.state.datasets.list();
        let summaries = datasets
            .iter()
            .map(|ds| proto::DatasetSummary {
                name: ds.name.clone(),
                num_items: ds.num_items as i32,
                num_item_features: ds.num_item_features as i32,
                memory_bytes: ds.memory_bytes as i64,
                loaded_at: ds.loaded_at.to_rfc3339(),
                score_request_count: ds.request_count() as i64,
            })
            .collect();

        Ok(Response::new(proto::ListDatasetsResponse {
            datasets: summaries,
        }))
    }

    async fn get_dataset_info(
        &self,
        request: Request<proto::DatasetInfoRequest>,
    ) -> Result<Response<proto::DatasetInfoResponse>, Status> {
        let name = &request.into_inner().name;
        let ds = self
            .state
            .datasets
            .get(name)
            .ok_or_else(|| Status::not_found(format!("dataset '{}' not found", name)))?;

        let ranges = ds
            .feature_ranges()
            .iter()
            .map(|&(idx, min, max)| proto::FeatureRange {
                feature_index: idx as i32,
                min_value: min,
                max_value: max,
            })
            .collect();

        Ok(Response::new(proto::DatasetInfoResponse {
            summary: Some(proto::DatasetSummary {
                name: ds.name.clone(),
                num_items: ds.num_items as i32,
                num_item_features: ds.num_item_features as i32,
                memory_bytes: ds.memory_bytes as i64,
                loaded_at: ds.loaded_at.to_rfc3339(),
                score_request_count: ds.request_count() as i64,
            }),
            feature_ranges: ranges,
        }))
    }
}

impl DatasetServiceImpl {
    fn update_dataset_metrics(&self, name: &str, num_items: usize, memory_bytes: usize) {
        self.state
            .metrics
            .dataset_items
            .with_label_values(&[name])
            .set(num_items as f64);
        self.state
            .metrics
            .dataset_memory_bytes
            .with_label_values(&[name])
            .set(memory_bytes as f64);
        self.state
            .metrics
            .loaded_datasets
            .set(self.state.datasets.count() as i64);
    }
}

#[cfg(test)]
mod service_state_helper {
    use super::*;
    use crate::engine::CpuScorer;
    use crate::ffi::ModelInfo;

    pub fn make_state() -> Arc<ServiceState> {
        let info = ModelInfo::query();
        Arc::new(ServiceState {
            scorer: Arc::new(CpuScorer::new()),
            datasets: Arc::new(DatasetManager::new(16, info.num_item_features)),
            metrics: Arc::new(Metrics::new()),
            start_time: Instant::now(),
        })
    }

    pub fn make_candidate_matrix(n_items: usize) -> Vec<f32> {
        let info = ModelInfo::query();
        (0..n_items * info.num_item_features)
            .map(|i| (i as f32 * 0.01) % 1.0)
            .collect()
    }
}

#[cfg(test)]
mod tests_dataset_service {
    use super::service_state_helper::*;
    use super::*;
    use crate::ffi::ModelInfo;
    use proto::dataset_service_server::DatasetService;

    #[tokio::test]
    async fn load_and_list() {
        let state = make_state();
        let svc = DatasetServiceImpl {
            state: state.clone(),
        };
        let info = ModelInfo::query();

        let resp = svc
            .load_dataset(Request::new(proto::LoadDatasetRequest {
                name: "test_ds".into(),
                candidate_matrix: make_candidate_matrix(100),
                num_items: 100,
                num_item_features: info.num_item_features as i32,
                replace_if_exists: false,
            }))
            .await
            .unwrap()
            .into_inner();

        assert!(resp.success);
        assert_eq!(resp.dataset_name, "test_ds");
        assert_eq!(resp.num_items, 100);
        assert!(resp.load_duration_us > 0);

        // List should contain the dataset
        let list = svc
            .list_datasets(Request::new(proto::ListDatasetsRequest {}))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(list.datasets.len(), 1);
        assert_eq!(list.datasets[0].name, "test_ds");
    }

    #[tokio::test]
    async fn load_bad_dimensions() {
        let state = make_state();
        let svc = DatasetServiceImpl { state };

        let result = svc
            .load_dataset(Request::new(proto::LoadDatasetRequest {
                name: "bad".into(),
                candidate_matrix: vec![1.0; 10],
                num_items: 5,
                num_item_features: 2, // wrong — model expects 26
                replace_if_exists: false,
            }))
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn unload() {
        let state = make_state();
        let svc = DatasetServiceImpl {
            state: state.clone(),
        };
        let info = ModelInfo::query();

        svc.load_dataset(Request::new(proto::LoadDatasetRequest {
            name: "to_unload".into(),
            candidate_matrix: make_candidate_matrix(10),
            num_items: 10,
            num_item_features: info.num_item_features as i32,
            replace_if_exists: false,
        }))
        .await
        .unwrap();

        let resp = svc
            .unload_dataset(Request::new(proto::UnloadDatasetRequest {
                name: "to_unload".into(),
            }))
            .await
            .unwrap()
            .into_inner();
        assert!(resp.success);

        // List should be empty
        let list = svc
            .list_datasets(Request::new(proto::ListDatasetsRequest {}))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(list.datasets.len(), 0);
    }

    #[tokio::test]
    async fn get_dataset_info() {
        let state = make_state();
        let svc = DatasetServiceImpl { state };
        let info = ModelInfo::query();

        svc.load_dataset(Request::new(proto::LoadDatasetRequest {
            name: "info_ds".into(),
            candidate_matrix: make_candidate_matrix(20),
            num_items: 20,
            num_item_features: info.num_item_features as i32,
            replace_if_exists: false,
        }))
        .await
        .unwrap();

        let resp = svc
            .get_dataset_info(Request::new(proto::DatasetInfoRequest {
                name: "info_ds".into(),
            }))
            .await
            .unwrap()
            .into_inner();

        let summary = resp.summary.unwrap();
        assert_eq!(summary.num_items, 20);
        assert_eq!(resp.feature_ranges.len(), info.num_item_features);
    }
}

#[cfg(test)]
mod tests_score_service {
    use super::service_state_helper::*;
    use super::*;
    use crate::ffi::ModelInfo;
    use proto::score_service_server::ScoreService;

    #[tokio::test]
    async fn score_topk() {
        let state = make_state();
        let info = ModelInfo::query();

        // Load dataset first
        state
            .datasets
            .load(
                "ads".into(),
                make_candidate_matrix(200),
                200,
                info.num_item_features,
                false,
            )
            .unwrap();

        let svc = ScoreServiceImpl { state };
        let user_vec: Vec<f32> = vec![0.5; info.num_user_features];

        let resp = svc
            .score_top_k(Request::new(proto::ScoreRequest {
                dataset_name: "ads".into(),
                user_vector: user_vec,
                top_k: 10,
                request_id: "req-1".into(),
            }))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(resp.candidates.len(), 10);
        assert_eq!(resp.items_scored, 200);
        assert_eq!(resp.backend, "cpu");
        assert_eq!(resp.request_id, "req-1");
        assert!(resp.latency_us >= 0);

        // Candidates should be descending
        for i in 1..resp.candidates.len() {
            assert!(resp.candidates[i - 1].score >= resp.candidates[i].score);
        }
    }

    #[tokio::test]
    async fn score_missing_dataset() {
        let state = make_state();
        let svc = ScoreServiceImpl { state };

        let result = svc
            .score_top_k(Request::new(proto::ScoreRequest {
                dataset_name: "nonexistent".into(),
                user_vector: vec![],
                top_k: 5,
                request_id: "".into(),
            }))
            .await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn score_batch() {
        let state = make_state();
        let info = ModelInfo::query();

        state
            .datasets
            .load(
                "batch_ds".into(),
                make_candidate_matrix(50),
                50,
                info.num_item_features,
                false,
            )
            .unwrap();

        let svc = ScoreServiceImpl { state };

        let resp = svc
            .score_batch(Request::new(proto::ScoreBatchRequest {
                dataset_name: "batch_ds".into(),
                users: vec![
                    proto::UserScoreRequest {
                        user_id: "u1".into(),
                        user_vector: vec![0.1; info.num_user_features],
                    },
                    proto::UserScoreRequest {
                        user_id: "u2".into(),
                        user_vector: vec![0.9; info.num_user_features],
                    },
                ],
                top_k: 5,
                request_id: "batch-1".into(),
            }))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(resp.results.len(), 2);
        assert_eq!(resp.results[0].user_id, "u1");
        assert_eq!(resp.results[1].user_id, "u2");
        assert_eq!(resp.results[0].candidates.len(), 5);
        assert!(resp.total_latency_us > 0);
    }
}

#[cfg(test)]
mod tests_stats_service {
    use super::service_state_helper::*;
    use super::*;
    use crate::ffi::ModelInfo;
    use proto::stats_service_server::StatsService;

    #[tokio::test]
    async fn healthz() {
        let state = make_state();
        let svc = StatsServiceImpl { state };

        let resp = svc
            .get_healthz(Request::new(proto::GetHealthzRequest {}))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(resp.status, proto::health_response::Status::Serving as i32);
        assert_eq!(resp.message, "ok");
    }

    #[tokio::test]
    async fn stats() {
        let state = make_state();
        let info = ModelInfo::query();

        // Load a dataset so stats reflect it
        state
            .datasets
            .load(
                "stats_ds".into(),
                (0..10 * info.num_item_features).map(|i| i as f32).collect(),
                10,
                info.num_item_features,
                false,
            )
            .unwrap();

        let svc = StatsServiceImpl { state };

        let resp = svc
            .get_stats(Request::new(proto::GetStatsRequest {}))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(resp.backend, "cpu");
        assert_eq!(resp.loaded_datasets, 1);
        assert!(resp.total_dataset_memory_bytes > 0);
        assert!(!resp.version.is_empty());
    }
}

// ============================================================
// StatsService
// ============================================================

pub struct StatsServiceImpl {
    pub state: Arc<ServiceState>,
}

#[tonic::async_trait]
impl proto::stats_service_server::StatsService for StatsServiceImpl {
    async fn get_stats(
        &self,
        _request: Request<proto::GetStatsRequest>,
    ) -> Result<Response<proto::StatsResponse>, Status> {
        let uptime = self.state.start_time.elapsed().as_secs() as i64;

        // Gather from prometheus metrics
        let families = self.state.metrics.registry.gather();
        let mut total_requests = 0i64;
        let mut total_items = 0i64;

        for family in &families {
            match family.get_name() {
                "score_requests_total" => {
                    for m in family.get_metric() {
                        total_requests += m.get_counter().get_value() as i64;
                    }
                }
                "items_scored_total" => {
                    for m in family.get_metric() {
                        total_items += m.get_counter().get_value() as i64;
                    }
                }
                _ => {}
            }
        }

        let avg_items_per_sec = if uptime > 0 {
            total_items as f64 / uptime as f64
        } else {
            0.0
        };

        Ok(Response::new(proto::StatsResponse {
            version: env!("CARGO_PKG_VERSION").to_string(),
            backend: self.state.scorer.backend_name().to_string(),
            uptime_seconds: uptime,
            total_score_requests: total_requests,
            total_items_scored: total_items,
            p50_latency_us: 0, // Would need HdrHistogram for accurate percentiles
            p95_latency_us: 0,
            p99_latency_us: 0,
            loaded_datasets: self.state.datasets.count() as i32,
            total_dataset_memory_bytes: self.state.datasets.total_memory_bytes() as i64,
            avg_items_per_second: avg_items_per_sec,
        }))
    }

    async fn get_healthz(
        &self,
        _request: Request<proto::GetHealthzRequest>,
    ) -> Result<Response<proto::HealthResponse>, Status> {
        let cuda_available = cfg!(feature = "cuda");
        let datasets_loaded = self.state.datasets.count() as i32;

        Ok(Response::new(proto::HealthResponse {
            status: proto::health_response::Status::Serving.into(),
            message: "ok".to_string(),
            cuda_available,
            datasets_loaded,
        }))
    }
}
