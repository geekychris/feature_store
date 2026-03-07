use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("dataset '{0}' not found")]
    NotFound(String),
    #[error("dataset '{0}' already exists (set replace_if_exists=true to overwrite)")]
    AlreadyExists(String),
    #[error("max datasets ({0}) reached")]
    MaxReached(usize),
    #[error("invalid dimensions: expected {expected} item features, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("data size mismatch: {0} floats cannot form {1} items of {2} features")]
    SizeMismatch(usize, usize, usize),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// A loaded CandidateMatrix with metadata.
#[derive(Debug)]
pub struct Dataset {
    pub name: String,
    /// Row-major float32: [num_items × num_item_features]
    pub candidate_matrix: Vec<f32>,
    pub num_items: usize,
    pub num_item_features: usize,
    pub memory_bytes: usize,
    pub loaded_at: DateTime<Utc>,
    pub score_request_count: AtomicU64,
}

impl Dataset {
    pub fn new(name: String, data: Vec<f32>, num_items: usize, num_item_features: usize) -> Self {
        let memory_bytes = data.len() * std::mem::size_of::<f32>();
        Self {
            name,
            candidate_matrix: data,
            num_items,
            num_item_features,
            memory_bytes,
            loaded_at: Utc::now(),
            score_request_count: AtomicU64::new(0),
        }
    }

    pub fn increment_requests(&self) {
        self.score_request_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn request_count(&self) -> u64 {
        self.score_request_count.load(Ordering::Relaxed)
    }

    /// Compute per-feature min/max ranges.
    pub fn feature_ranges(&self) -> Vec<(usize, f32, f32)> {
        let mut ranges = Vec::with_capacity(self.num_item_features);
        for f in 0..self.num_item_features {
            let mut min = f32::MAX;
            let mut max = f32::MIN;
            for i in 0..self.num_items {
                let val = self.candidate_matrix[i * self.num_item_features + f];
                if val < min {
                    min = val;
                }
                if val > max {
                    max = val;
                }
            }
            ranges.push((f, min, max));
        }
        ranges
    }
}

/// Manages named datasets with concurrent access.
pub struct DatasetManager {
    datasets: DashMap<String, Arc<Dataset>>,
    max_datasets: usize,
    expected_item_features: usize,
}

impl DatasetManager {
    pub fn new(max_datasets: usize, expected_item_features: usize) -> Self {
        Self {
            datasets: DashMap::new(),
            max_datasets,
            expected_item_features,
        }
    }

    /// Load a dataset from a flat float buffer.
    pub fn load(
        &self,
        name: String,
        data: Vec<f32>,
        num_items: usize,
        num_item_features: usize,
        replace: bool,
    ) -> Result<Arc<Dataset>, DatasetError> {
        // Validate dimensions
        if num_item_features != self.expected_item_features {
            return Err(DatasetError::DimensionMismatch {
                expected: self.expected_item_features,
                actual: num_item_features,
            });
        }
        let expected_len = num_items * num_item_features;
        if data.len() != expected_len {
            return Err(DatasetError::SizeMismatch(
                data.len(),
                num_items,
                num_item_features,
            ));
        }

        // Check capacity
        if !replace && self.datasets.contains_key(&name) {
            return Err(DatasetError::AlreadyExists(name));
        }
        if !self.datasets.contains_key(&name) && self.datasets.len() >= self.max_datasets {
            return Err(DatasetError::MaxReached(self.max_datasets));
        }

        let ds = Arc::new(Dataset::new(
            name.clone(),
            data,
            num_items,
            num_item_features,
        ));
        self.datasets.insert(name, ds.clone());
        Ok(ds)
    }

    /// Load from a binary file (row-major float32).
    pub fn load_from_file(
        &self,
        name: String,
        path: &str,
        num_item_features: usize,
        replace: bool,
    ) -> Result<Arc<Dataset>, DatasetError> {
        let bytes = std::fs::read(path)?;
        let num_floats = bytes.len() / std::mem::size_of::<f32>();
        let num_items = num_floats / num_item_features;

        if num_items == 0 || num_floats != num_items * num_item_features {
            return Err(DatasetError::SizeMismatch(
                num_floats,
                num_items,
                num_item_features,
            ));
        }

        // Reinterpret bytes as f32 (assume native endian)
        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        self.load(name, data, num_items, num_item_features, replace)
    }

    pub fn get(&self, name: &str) -> Option<Arc<Dataset>> {
        self.datasets.get(name).map(|r| r.value().clone())
    }

    pub fn unload(&self, name: &str) -> Result<(), DatasetError> {
        self.datasets
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| DatasetError::NotFound(name.to_string()))
    }

    pub fn list(&self) -> Vec<Arc<Dataset>> {
        self.datasets.iter().map(|r| r.value().clone()).collect()
    }

    pub fn count(&self) -> usize {
        self.datasets.len()
    }

    pub fn total_memory_bytes(&self) -> usize {
        self.datasets.iter().map(|r| r.value().memory_bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(num_items: usize, num_features: usize) -> Vec<f32> {
        (0..num_items * num_features)
            .map(|i| i as f32 * 0.1)
            .collect()
    }

    #[test]
    fn load_and_get() {
        let mgr = DatasetManager::new(4, 26);
        let data = make_data(100, 26);
        let ds = mgr
            .load("test".into(), data.clone(), 100, 26, false)
            .unwrap();
        assert_eq!(ds.name, "test");
        assert_eq!(ds.num_items, 100);
        assert_eq!(ds.num_item_features, 26);
        assert_eq!(ds.memory_bytes, 100 * 26 * 4);

        let fetched = mgr.get("test").unwrap();
        assert_eq!(fetched.num_items, 100);
    }

    #[test]
    fn get_missing_returns_none() {
        let mgr = DatasetManager::new(4, 26);
        assert!(mgr.get("nope").is_none());
    }

    #[test]
    fn load_duplicate_without_replace_fails() {
        let mgr = DatasetManager::new(4, 26);
        mgr.load("ds".into(), make_data(10, 26), 10, 26, false)
            .unwrap();
        let err = mgr
            .load("ds".into(), make_data(10, 26), 10, 26, false)
            .unwrap_err();
        assert!(matches!(err, DatasetError::AlreadyExists(_)));
    }

    #[test]
    fn load_duplicate_with_replace_succeeds() {
        let mgr = DatasetManager::new(4, 26);
        mgr.load("ds".into(), make_data(10, 26), 10, 26, false)
            .unwrap();
        let ds = mgr
            .load("ds".into(), make_data(20, 26), 20, 26, true)
            .unwrap();
        assert_eq!(ds.num_items, 20);
        assert_eq!(mgr.count(), 1);
    }

    #[test]
    fn dimension_mismatch() {
        let mgr = DatasetManager::new(4, 26);
        let err = mgr
            .load("ds".into(), make_data(10, 13), 10, 13, false)
            .unwrap_err();
        assert!(matches!(
            err,
            DatasetError::DimensionMismatch {
                expected: 26,
                actual: 13
            }
        ));
    }

    #[test]
    fn size_mismatch() {
        let mgr = DatasetManager::new(4, 26);
        // 100 floats but claim 10 items x 26 features = 260
        let err = mgr
            .load("ds".into(), make_data(100, 1), 10, 26, false)
            .unwrap_err();
        assert!(matches!(err, DatasetError::SizeMismatch(..)));
    }

    #[test]
    fn max_capacity_reached() {
        let mgr = DatasetManager::new(2, 26);
        mgr.load("a".into(), make_data(1, 26), 1, 26, false)
            .unwrap();
        mgr.load("b".into(), make_data(1, 26), 1, 26, false)
            .unwrap();
        let err = mgr
            .load("c".into(), make_data(1, 26), 1, 26, false)
            .unwrap_err();
        assert!(matches!(err, DatasetError::MaxReached(2)));
    }

    #[test]
    fn unload() {
        let mgr = DatasetManager::new(4, 26);
        mgr.load("ds".into(), make_data(5, 26), 5, 26, false)
            .unwrap();
        assert_eq!(mgr.count(), 1);
        mgr.unload("ds").unwrap();
        assert_eq!(mgr.count(), 0);
        assert!(mgr.get("ds").is_none());
    }

    #[test]
    fn unload_missing_fails() {
        let mgr = DatasetManager::new(4, 26);
        let err = mgr.unload("nope").unwrap_err();
        assert!(matches!(err, DatasetError::NotFound(_)));
    }

    #[test]
    fn list_datasets() {
        let mgr = DatasetManager::new(4, 26);
        mgr.load("a".into(), make_data(1, 26), 1, 26, false)
            .unwrap();
        mgr.load("b".into(), make_data(2, 26), 2, 26, false)
            .unwrap();
        let listed = mgr.list();
        assert_eq!(listed.len(), 2);
        let mut names: Vec<_> = listed.iter().map(|d| d.name.as_str()).collect();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn total_memory_bytes() {
        let mgr = DatasetManager::new(4, 26);
        mgr.load("a".into(), make_data(10, 26), 10, 26, false)
            .unwrap();
        mgr.load("b".into(), make_data(20, 26), 20, 26, false)
            .unwrap();
        assert_eq!(mgr.total_memory_bytes(), (10 + 20) * 26 * 4);
    }

    #[test]
    fn feature_ranges() {
        // 3 items x 2 features (using expected=2 for this test)
        let mgr = DatasetManager::new(4, 2);
        let data = vec![1.0, 5.0, 3.0, 2.0, 7.0, 1.0];
        let ds = mgr.load("r".into(), data, 3, 2, false).unwrap();
        let ranges = ds.feature_ranges();
        assert_eq!(ranges.len(), 2);
        // Feature 0: [1.0, 3.0, 7.0] -> min=1.0, max=7.0
        assert_eq!(ranges[0], (0, 1.0, 7.0));
        // Feature 1: [5.0, 2.0, 1.0] -> min=1.0, max=5.0
        assert_eq!(ranges[1], (1, 1.0, 5.0));
    }

    #[test]
    fn request_counter() {
        let ds = Dataset::new("test".into(), make_data(1, 26), 1, 26);
        assert_eq!(ds.request_count(), 0);
        ds.increment_requests();
        ds.increment_requests();
        assert_eq!(ds.request_count(), 2);
    }

    #[test]
    fn load_from_binary_file() {
        let mgr = DatasetManager::new(4, 2);
        let items: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // 2 items x 2 features
        let bytes: Vec<u8> = items.iter().flat_map(|f| f.to_le_bytes()).collect();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &bytes).unwrap();

        let ds = mgr
            .load_from_file("file_ds".into(), tmp.path().to_str().unwrap(), 2, false)
            .unwrap();
        assert_eq!(ds.num_items, 2);
        assert_eq!(ds.num_item_features, 2);
        assert_eq!(ds.candidate_matrix, items);
    }
}
