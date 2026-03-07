//! FFI bindings to the auto-generated C/CUDA GBDT scoring library.
//!
//! These functions are compiled from `scoring_split_core.h` via the shim in `ffi/scoring_shim.c`.
//! The generated code is never modified — only consumed through this interface.

use std::os::raw::{c_float, c_int};

extern "C" {
    /// Score all items against one user on CPU.
    /// - `user_features`: [NUM_USER_FEATURES] float32
    /// - `item_features`: [n_items × NUM_ITEM_FEATURES] row-major float32
    /// - `scores`: [n_items] output float32
    pub fn scorched_score_user_cpu(
        user_features: *const c_float,
        item_features: *const c_float,
        scores: *mut c_float,
        n_items: c_int,
    );

    /// Score all items + return top-K sorted descending.
    /// Returns the number of valid results (min(k, n_items)), or -1 on error.
    pub fn scorched_score_topk_cpu(
        user_features: *const c_float,
        item_features: *const c_float,
        n_items: c_int,
        k: c_int,
        out_indices: *mut c_int,
        out_scores: *mut c_float,
    ) -> c_int;

    // Model constants
    pub fn scorched_num_features() -> c_int;
    pub fn scorched_num_user_features() -> c_int;
    pub fn scorched_num_item_features() -> c_int;
    pub fn scorched_num_trees() -> c_int;
    pub fn scorched_base_score() -> c_float;
}

// ---- CUDA bindings (only compiled with --features cuda) ----

#[cfg(feature = "cuda")]
extern "C" {
    pub fn scorched_cuda_context_create(max_items: c_int) -> *mut std::ffi::c_void;
    pub fn scorched_cuda_context_destroy(ctx: *mut std::ffi::c_void);
    pub fn scorched_cuda_load_items(
        ctx: *mut std::ffi::c_void,
        items: *const c_float,
        n_items: c_int,
    ) -> c_int;
    pub fn scorched_cuda_score_user(
        ctx: *mut std::ffi::c_void,
        user: *const c_float,
        scores: *mut c_float,
    ) -> c_int;
    pub fn scorched_cuda_score_topk(
        ctx: *mut std::ffi::c_void,
        user: *const c_float,
        k: c_int,
        out_indices: *mut c_int,
        out_scores: *mut c_float,
    ) -> c_int;
    pub fn scorched_cuda_items_loaded(ctx: *mut std::ffi::c_void) -> c_int;
}

/// Safe wrapper to query model dimensions at runtime.
pub struct ModelInfo {
    pub num_features: usize,
    pub num_user_features: usize,
    pub num_item_features: usize,
    pub num_trees: usize,
    pub base_score: f32,
}

impl ModelInfo {
    pub fn query() -> Self {
        unsafe {
            Self {
                num_features: scorched_num_features() as usize,
                num_user_features: scorched_num_user_features() as usize,
                num_item_features: scorched_num_item_features() as usize,
                num_trees: scorched_num_trees() as usize,
                base_score: scorched_base_score(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_info_has_valid_constants() {
        let info = ModelInfo::query();
        assert!(info.num_trees > 0, "model should have at least one tree");
        assert_eq!(
            info.num_features,
            info.num_user_features + info.num_item_features
        );
        assert!(info.num_user_features > 0);
        assert!(info.num_item_features > 0);
    }

    #[test]
    fn model_info_matches_expected_dimensions() {
        // The generated model: 300 trees, 13 user + 26 item = 39 total
        let info = ModelInfo::query();
        assert_eq!(info.num_trees, 300);
        assert_eq!(info.num_features, 39);
        assert_eq!(info.num_user_features, 13);
        assert_eq!(info.num_item_features, 26);
    }

    #[test]
    fn score_user_cpu_returns_scores() {
        let info = ModelInfo::query();
        let user = vec![0.5f32; info.num_user_features];
        let n_items = 3;
        let items = vec![0.1f32; n_items * info.num_item_features];
        let mut scores = vec![0.0f32; n_items];

        unsafe {
            scorched_score_user_cpu(
                user.as_ptr(),
                items.as_ptr(),
                scores.as_mut_ptr(),
                n_items as i32,
            );
        }

        // Scores should be sigmoid outputs in (0,1)
        for &s in &scores {
            assert!(s > 0.0 && s < 1.0, "score {} not in (0,1)", s);
        }
        // All identical items should produce identical scores
        assert_eq!(scores[0], scores[1]);
        assert_eq!(scores[1], scores[2]);
    }

    #[test]
    fn topk_returns_sorted_descending() {
        let info = ModelInfo::query();
        let user = vec![0.3f32; info.num_user_features];
        let n_items = 50;
        // Vary item features so scores differ
        let items: Vec<f32> = (0..n_items * info.num_item_features)
            .map(|i| (i as f32 * 0.01) % 1.0)
            .collect();
        let k = 10;
        let mut out_idx = vec![0i32; k];
        let mut out_scores = vec![0.0f32; k];

        let result = unsafe {
            scorched_score_topk_cpu(
                user.as_ptr(),
                items.as_ptr(),
                n_items as i32,
                k as i32,
                out_idx.as_mut_ptr(),
                out_scores.as_mut_ptr(),
            )
        };

        assert_eq!(result, k as i32);
        // Verify descending order
        for i in 1..k {
            assert!(
                out_scores[i - 1] >= out_scores[i],
                "scores not descending: [{}]={} < [{}]={}",
                i - 1,
                out_scores[i - 1],
                i,
                out_scores[i]
            );
        }
        // All indices should be valid
        for &idx in &out_idx {
            assert!(idx >= 0 && idx < n_items as i32);
        }
    }
}
