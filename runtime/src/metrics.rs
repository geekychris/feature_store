use std::sync::Arc;
use std::time::Instant;

use prometheus::{
    Encoder, GaugeVec, HistogramOpts, HistogramVec, IntCounterVec, IntGauge, Opts, Registry,
    TextEncoder,
};

/// All Prometheus metrics for the scoring runtime.
#[derive(Clone)]
#[allow(dead_code)]
pub struct Metrics {
    pub registry: Registry,
    pub score_duration: HistogramVec,
    pub score_requests: IntCounterVec,
    pub items_scored: IntCounterVec,
    pub dataset_load_duration: HistogramVec,
    pub loaded_datasets: IntGauge,
    pub dataset_items: GaugeVec,
    pub dataset_memory_bytes: GaugeVec,
    pub cuda_available: IntGauge,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();

        let score_duration = HistogramVec::new(
            HistogramOpts::new(
                "score_request_duration_seconds",
                "Scoring request latency in seconds",
            )
            .buckets(vec![
                0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0,
            ]),
            &["dataset", "method", "backend"],
        )
        .unwrap();

        let score_requests = IntCounterVec::new(
            Opts::new("score_requests_total", "Total scoring requests"),
            &["dataset", "status"],
        )
        .unwrap();

        let items_scored = IntCounterVec::new(
            Opts::new("items_scored_total", "Total items scored"),
            &["dataset"],
        )
        .unwrap();

        let dataset_load_duration = HistogramVec::new(
            HistogramOpts::new(
                "dataset_load_duration_seconds",
                "Dataset load latency in seconds",
            )
            .buckets(vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0]),
            &["dataset"],
        )
        .unwrap();

        let loaded_datasets =
            IntGauge::new("loaded_datasets", "Number of loaded datasets").unwrap();

        let dataset_items = GaugeVec::new(
            Opts::new("dataset_items", "Number of items per dataset"),
            &["dataset"],
        )
        .unwrap();

        let dataset_memory_bytes = GaugeVec::new(
            Opts::new("dataset_memory_bytes", "Memory usage per dataset in bytes"),
            &["dataset"],
        )
        .unwrap();

        let cuda_available =
            IntGauge::new("cuda_available", "Whether CUDA backend is available").unwrap();

        registry.register(Box::new(score_duration.clone())).unwrap();
        registry.register(Box::new(score_requests.clone())).unwrap();
        registry.register(Box::new(items_scored.clone())).unwrap();
        registry
            .register(Box::new(dataset_load_duration.clone()))
            .unwrap();
        registry
            .register(Box::new(loaded_datasets.clone()))
            .unwrap();
        registry.register(Box::new(dataset_items.clone())).unwrap();
        registry
            .register(Box::new(dataset_memory_bytes.clone()))
            .unwrap();
        registry.register(Box::new(cuda_available.clone())).unwrap();

        Self {
            registry,
            score_duration,
            score_requests,
            items_scored,
            dataset_load_duration,
            loaded_datasets,
            dataset_items,
            dataset_memory_bytes,
            cuda_available,
        }
    }

    /// Encode all metrics as Prometheus text format.
    pub fn encode(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }
}

/// RAII timer for scoring latency.
pub struct ScoringTimer {
    start: Instant,
    metrics: Arc<Metrics>,
    dataset: String,
    method: String,
    backend: String,
}

impl ScoringTimer {
    pub fn new(metrics: Arc<Metrics>, dataset: &str, method: &str, backend: &str) -> Self {
        Self {
            start: Instant::now(),
            metrics,
            dataset: dataset.to_string(),
            method: method.to_string(),
            backend: backend.to_string(),
        }
    }

    pub fn elapsed_us(&self) -> i64 {
        self.start.elapsed().as_micros() as i64
    }
}

impl Drop for ScoringTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        self.metrics
            .score_duration
            .with_label_values(&[&self.dataset, &self.method, &self.backend])
            .observe(elapsed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_new_does_not_panic() {
        let m = Metrics::new();
        // All metrics should be registered and queryable
        assert!(m.loaded_datasets.get() == 0);
        assert!(m.cuda_available.get() == 0);
    }

    #[test]
    fn encode_returns_valid_prometheus_text() {
        let m = Metrics::new();
        m.score_requests.with_label_values(&["test_ds", "ok"]).inc();
        m.loaded_datasets.set(2);

        let text = m.encode();
        assert!(text.contains("score_requests_total"));
        assert!(text.contains("loaded_datasets 2"));
        assert!(text.contains("test_ds"));
    }

    #[test]
    fn scoring_timer_records_on_drop() {
        let m = Arc::new(Metrics::new());
        {
            let _t = ScoringTimer::new(m.clone(), "ds", "score_topk", "cpu");
            std::thread::sleep(std::time::Duration::from_millis(5));
            assert!(_t.elapsed_us() > 0);
        }
        // After drop, histogram should have an observation
        let text = m.encode();
        assert!(text.contains("score_request_duration_seconds"));
    }
}

/// Start the Prometheus HTTP metrics server on the given port.
pub async fn serve_metrics(metrics: Arc<Metrics>, port: u16) {
    use http_body_util::Full;
    use hyper::body::Bytes;
    use hyper::service::service_fn;
    use hyper::{Request, Response};
    use hyper_util::rt::TokioIo;
    use tokio::net::TcpListener;

    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).await.unwrap();
    tracing::info!("Prometheus metrics server listening on {}", addr);

    loop {
        let (stream, _) = match listener.accept().await {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("Metrics server accept error: {}", e);
                continue;
            }
        };

        let metrics = metrics.clone();
        tokio::spawn(async move {
            let service = service_fn(move |_req: Request<hyper::body::Incoming>| {
                let body = metrics.encode();
                async move { Ok::<_, hyper::Error>(Response::new(Full::new(Bytes::from(body)))) }
            });
            let io = TokioIo::new(stream);
            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, service)
                .await
            {
                tracing::debug!("Metrics connection error: {}", e);
            }
        });
    }
}
