use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(name = "scorched", version, about = "GBDT scoring runtime")]
pub struct Config {
    /// gRPC listen port
    #[arg(long, default_value = "50051", env = "SCORCHED_PORT")]
    pub port: u16,

    /// Prometheus metrics HTTP port
    #[arg(long, default_value = "9091", env = "SCORCHED_METRICS_PORT")]
    pub metrics_port: u16,

    /// Enable CUDA backend (requires --features cuda at build time)
    #[arg(long, default_value = "false", env = "SCORCHED_CUDA")]
    pub cuda: bool,

    /// Maximum number of named datasets
    #[arg(long, default_value = "16", env = "SCORCHED_MAX_DATASETS")]
    pub max_datasets: usize,

    /// Number of worker threads for CPU scoring (0 = num_cpus)
    #[arg(long, default_value = "0", env = "SCORCHED_WORKER_THREADS")]
    pub worker_threads: usize,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info", env = "SCORCHED_LOG_LEVEL")]
    pub log_level: String,

    /// gRPC max receive message size in bytes (default 64MB for large datasets)
    #[arg(long, default_value = "67108864", env = "SCORCHED_MAX_MSG_SIZE")]
    pub max_message_size: usize,

    /// Request timeout in seconds
    #[arg(long, default_value = "30", env = "SCORCHED_TIMEOUT")]
    pub timeout_seconds: u64,

    /// OpenTelemetry OTLP endpoint (empty = disabled)
    #[arg(long, default_value = "", env = "OTEL_EXPORTER_OTLP_ENDPOINT")]
    pub otlp_endpoint: String,

    /// Enable JSON structured logging
    #[arg(long, default_value = "false", env = "SCORCHED_JSON_LOG")]
    pub json_log: bool,
}

impl Config {
    pub fn effective_worker_threads(&self) -> usize {
        if self.worker_threads == 0 {
            num_cpus::get()
        } else {
            self.worker_threads
        }
    }

    pub fn backend_name(&self) -> &'static str {
        if self.cuda {
            "cuda"
        } else {
            "cpu"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> Config {
        Config {
            port: 50051,
            metrics_port: 9091,
            cuda: false,
            max_datasets: 16,
            worker_threads: 0,
            log_level: "info".into(),
            max_message_size: 67108864,
            timeout_seconds: 30,
            otlp_endpoint: "".into(),
            json_log: false,
        }
    }

    #[test]
    fn backend_name_cpu() {
        let cfg = default_config();
        assert_eq!(cfg.backend_name(), "cpu");
    }

    #[test]
    fn backend_name_cuda() {
        let mut cfg = default_config();
        cfg.cuda = true;
        assert_eq!(cfg.backend_name(), "cuda");
    }

    #[test]
    fn effective_workers_zero_uses_num_cpus() {
        let cfg = default_config();
        assert!(cfg.effective_worker_threads() > 0);
    }

    #[test]
    fn effective_workers_explicit() {
        let mut cfg = default_config();
        cfg.worker_threads = 4;
        assert_eq!(cfg.effective_worker_threads(), 4);
    }
}
