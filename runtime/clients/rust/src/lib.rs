pub mod proto {
    tonic::include_proto!("scorched");
}

use anyhow::Result;
use tonic::transport::Channel;

pub use proto::*;

/// High-level client for the scorched scoring runtime.
pub struct ScoringClient {
    score: proto::score_service_client::ScoreServiceClient<Channel>,
    dataset: proto::dataset_service_client::DatasetServiceClient<Channel>,
    stats: proto::stats_service_client::StatsServiceClient<Channel>,
}

impl ScoringClient {
    /// Connect to a scorched runtime instance.
    pub async fn connect(addr: &str) -> Result<Self> {
        let channel = Channel::from_shared(addr.to_string())?
            .connect()
            .await?;

        Ok(Self {
            score: proto::score_service_client::ScoreServiceClient::new(channel.clone()),
            dataset: proto::dataset_service_client::DatasetServiceClient::new(channel.clone()),
            stats: proto::stats_service_client::StatsServiceClient::new(channel),
        })
    }

    /// Score a user against a named dataset, returning top-K candidates.
    pub async fn score_topk(
        &mut self,
        dataset_name: &str,
        user_vector: Vec<f32>,
        top_k: i32,
    ) -> Result<ScoreResponse> {
        let resp = self
            .score
            .score_top_k(ScoreRequest {
                dataset_name: dataset_name.to_string(),
                user_vector,
                top_k,
                request_id: String::new(),
            })
            .await?;
        Ok(resp.into_inner())
    }

    /// Load a dataset from a flat float buffer.
    pub async fn load_dataset(
        &mut self,
        name: &str,
        candidate_matrix: Vec<f32>,
        num_items: i32,
        num_item_features: i32,
        replace: bool,
    ) -> Result<LoadDatasetResponse> {
        let resp = self
            .dataset
            .load_dataset(LoadDatasetRequest {
                name: name.to_string(),
                candidate_matrix,
                num_items,
                num_item_features,
                replace_if_exists: replace,
            })
            .await?;
        Ok(resp.into_inner())
    }

    /// Load a dataset from a server-local file.
    pub async fn load_dataset_from_file(
        &mut self,
        name: &str,
        file_path: &str,
        num_item_features: i32,
        replace: bool,
    ) -> Result<LoadDatasetResponse> {
        let resp = self
            .dataset
            .load_dataset_from_file(LoadDatasetFromFileRequest {
                name: name.to_string(),
                file_path: file_path.to_string(),
                num_item_features,
                replace_if_exists: replace,
                format: "binary".to_string(),
            })
            .await?;
        Ok(resp.into_inner())
    }

    /// List all loaded datasets.
    pub async fn list_datasets(&mut self) -> Result<ListDatasetsResponse> {
        let resp = self
            .dataset
            .list_datasets(ListDatasetsRequest {})
            .await?;
        Ok(resp.into_inner())
    }

    /// Get runtime stats.
    pub async fn get_stats(&mut self) -> Result<StatsResponse> {
        let resp = self
            .stats
            .get_stats(GetStatsRequest {})
            .await?;
        Ok(resp.into_inner())
    }

    /// Health check.
    pub async fn healthz(&mut self) -> Result<HealthResponse> {
        let resp = self
            .stats
            .get_healthz(GetHealthzRequest {})
            .await?;
        Ok(resp.into_inner())
    }
}
