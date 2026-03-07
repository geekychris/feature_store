use std::sync::Arc;

use crate::dataset::Dataset;
use crate::ffi;

/// A scored result from the ranking engine.
#[derive(Debug, Clone)]
pub struct ScoredItem {
    pub index: i32,
    pub score: f32,
}

/// Backend-agnostic scoring interface.
pub trait Scorer: Send + Sync {
    fn backend_name(&self) -> &'static str;

    /// Score all items and return top-K sorted descending.
    fn score_topk(
        &self,
        user_features: &[f32],
        dataset: &Dataset,
        k: usize,
    ) -> Result<Vec<ScoredItem>, ScoringError>;

    /// Score all items and return all scores.
    fn score_all(&self, user_features: &[f32], dataset: &Dataset)
        -> Result<Vec<f32>, ScoringError>;
}

#[derive(Debug, thiserror::Error)]
#[allow(dead_code)]
pub enum ScoringError {
    #[error("invalid user vector length: expected {expected}, got {actual}")]
    UserVectorLength { expected: usize, actual: usize },
    #[error("k must be > 0")]
    InvalidK,
    #[error("FFI error: {0}")]
    Ffi(String),
    #[error("CUDA not available")]
    CudaNotAvailable,
}

/// CPU backend — calls into the C scoring library via FFI.
pub struct CpuScorer {
    num_user_features: usize,
}

impl CpuScorer {
    pub fn new() -> Self {
        let info = ffi::ModelInfo::query();
        Self {
            num_user_features: info.num_user_features,
        }
    }
}

impl Scorer for CpuScorer {
    fn backend_name(&self) -> &'static str {
        "cpu"
    }

    fn score_topk(
        &self,
        user_features: &[f32],
        dataset: &Dataset,
        k: usize,
    ) -> Result<Vec<ScoredItem>, ScoringError> {
        if k == 0 {
            return Err(ScoringError::InvalidK);
        }
        let user = self.resolve_user_vector(user_features)?;

        let mut out_indices = vec![0i32; k];
        let mut out_scores = vec![0.0f32; k];

        let result_k = unsafe {
            ffi::scorched_score_topk_cpu(
                user.as_ptr(),
                dataset.candidate_matrix.as_ptr(),
                dataset.num_items as i32,
                k as i32,
                out_indices.as_mut_ptr(),
                out_scores.as_mut_ptr(),
            )
        };

        if result_k < 0 {
            return Err(ScoringError::Ffi("score_topk_cpu returned -1".into()));
        }

        let n = result_k as usize;
        Ok((0..n)
            .map(|i| ScoredItem {
                index: out_indices[i],
                score: out_scores[i],
            })
            .collect())
    }

    fn score_all(
        &self,
        user_features: &[f32],
        dataset: &Dataset,
    ) -> Result<Vec<f32>, ScoringError> {
        let user = self.resolve_user_vector(user_features)?;
        let mut scores = vec![0.0f32; dataset.num_items];

        unsafe {
            ffi::scorched_score_user_cpu(
                user.as_ptr(),
                dataset.candidate_matrix.as_ptr(),
                scores.as_mut_ptr(),
                dataset.num_items as i32,
            );
        }

        Ok(scores)
    }
}

impl CpuScorer {
    /// Resolve the user vector: if empty, use zeros; if wrong length, error.
    fn resolve_user_vector(&self, user_features: &[f32]) -> Result<Vec<f32>, ScoringError> {
        if user_features.is_empty() {
            Ok(vec![0.0f32; self.num_user_features])
        } else if user_features.len() != self.num_user_features {
            Err(ScoringError::UserVectorLength {
                expected: self.num_user_features,
                actual: user_features.len(),
            })
        } else {
            Ok(user_features.to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;

    fn make_dataset(n_items: usize) -> Dataset {
        let info = ffi::ModelInfo::query();
        let data: Vec<f32> = (0..n_items * info.num_item_features)
            .map(|i| (i as f32 * 0.01) % 1.0)
            .collect();
        Dataset::new("test".into(), data, n_items, info.num_item_features)
    }

    #[test]
    fn cpu_scorer_topk() {
        let scorer = CpuScorer::new();
        let ds = make_dataset(100);
        let info = ffi::ModelInfo::query();
        let user = vec![0.5f32; info.num_user_features];

        let results = scorer.score_topk(&user, &ds, 5).unwrap();
        assert_eq!(results.len(), 5);
        // Descending order
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }
    }

    #[test]
    fn cpu_scorer_topk_k_greater_than_items() {
        let scorer = CpuScorer::new();
        let ds = make_dataset(3);
        let info = ffi::ModelInfo::query();
        let user = vec![0.5f32; info.num_user_features];

        let results = scorer.score_topk(&user, &ds, 100).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn cpu_scorer_topk_invalid_k() {
        let scorer = CpuScorer::new();
        let ds = make_dataset(10);
        let err = scorer.score_topk(&[], &ds, 0).unwrap_err();
        assert!(matches!(err, ScoringError::InvalidK));
    }

    #[test]
    fn cpu_scorer_score_all() {
        let scorer = CpuScorer::new();
        let ds = make_dataset(50);
        let info = ffi::ModelInfo::query();
        let user = vec![0.5f32; info.num_user_features];

        let scores = scorer.score_all(&user, &ds).unwrap();
        assert_eq!(scores.len(), 50);
        for &s in &scores {
            assert!(s > 0.0 && s < 1.0, "score {} not in (0,1)", s);
        }
    }

    #[test]
    fn cpu_scorer_empty_user_vector_uses_zeros() {
        let scorer = CpuScorer::new();
        let ds = make_dataset(10);
        // Empty user vector should work (defaults to zero)
        let results = scorer.score_topk(&[], &ds, 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn cpu_scorer_wrong_user_vector_length() {
        let scorer = CpuScorer::new();
        let ds = make_dataset(10);
        // Wrong length user vector
        let err = scorer.score_topk(&[1.0, 2.0], &ds, 3).unwrap_err();
        assert!(matches!(err, ScoringError::UserVectorLength { .. }));
    }

    #[test]
    fn create_scorer_returns_cpu() {
        let s = create_scorer(false);
        assert_eq!(s.backend_name(), "cpu");
    }
}

/// Creates the appropriate scorer based on configuration.
pub fn create_scorer(use_cuda: bool) -> Arc<dyn Scorer> {
    if use_cuda {
        #[cfg(feature = "cuda")]
        {
            tracing::info!("CUDA backend requested — initializing");
            // TODO: CudaScorer implementation wrapping ScoringContext
            tracing::warn!("CUDA scorer not yet wired — falling back to CPU");
            Arc::new(CpuScorer::new())
        }
        #[cfg(not(feature = "cuda"))]
        {
            tracing::warn!("CUDA requested but not compiled in — using CPU backend");
            Arc::new(CpuScorer::new())
        }
    } else {
        tracing::info!("Using CPU scoring backend");
        Arc::new(CpuScorer::new())
    }
}
