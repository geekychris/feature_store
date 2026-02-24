# CUDA Code Generator for XGBoost Models

Generates standalone CUDA or C source code from trained XGBoost tree ensemble
models. The generated code scores batches of feature vectors in parallel on
GPU (CUDA) or sequentially on CPU (C), with zero runtime dependencies beyond
CUDA or standard C math.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Code Generation Pipeline                      │
│                                                                  │
│  XGBoost Model (.ubj)                                           │
│       │                                                          │
│       ▼                                                          │
│  CudaCodeGenerator (Python)                                      │
│       │                                                          │
│       ├── Load model via xgboost.Booster                        │
│       ├── Extract trees as JSON (get_dump)                      │
│       ├── Calibrate base_score against raw margin output        │
│       ├── Emit tree functions (nested if/else)                  │
│       ├── Emit scoring kernel / CPU loop                        │
│       └── Emit host API + standalone main()                     │
│       │                                                          │
│       ▼                                                          │
│  Generated Source (.cu / .c)                                     │
│       │                                                          │
│       ├── nvcc ──► CUDA binary (GPU scoring)                    │
│       └── gcc  ──► C binary    (CPU scoring / verification)     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Generated Code Structure

Each generated file is self-contained and includes:

1. **Tree functions** — One `static inline` (C) or `__device__ __forceinline__`
   (CUDA) function per tree. Each is a nested if/else tree traversal with
   thresholds and leaf values baked in as constants.

2. **`score_all_trees()`** — Calls all tree functions and sums leaf values.

3. **Scoring kernel** (CUDA) or **`score_batch_cpu()`** (C) — The CUDA version
   launches one thread per sample. Each thread independently traverses all
   trees and computes the final probability. The CPU version is a simple loop.

4. **Host API** (CUDA only) — `ScoringContext` struct with pre-allocated device
   memory, `scoring_context_create()`, `scoring_context_destroy()`, and
   `score_batch()` for managing GPU memory and kernel launches.

5. **`main()`** — Standalone entry point that reads feature vectors from a
   binary file, scores them, and writes results to a binary file or stdout.

### Key Design Decisions

**Inlined trees vs. data-driven traversal**
Each tree is compiled as nested if/else statements. This means the model
structure is embedded in the instruction stream rather than loaded from memory
at runtime. For models up to ~1000 trees of depth ≤ 8, this produces faster
code because the CPU/GPU branch predictor can optimize the paths, and there
are no pointer-chasing memory accesses. For very large models (10,000+ trees),
a data-driven approach (tree nodes in a flat array) may be better to avoid
instruction cache pressure.

**NaN handling (missing values)**
XGBoost routes missing/NaN values to a learned direction at each split node.
Rather than checking `isnan()` explicitly, the generator uses IEEE 754
comparison semantics:
- When NaN routes the same way as "yes" (f < threshold):
  `if (!(f[i] >= threshold))` — true for both `f < threshold` and NaN
- When NaN routes the same way as "no" (f >= threshold):
  `if (f[i] < threshold)` — false for NaN, so it falls to else
- Only in the rare case where NaN goes a third direction is `isnan()` used

**Base score calibration**
Different XGBoost versions store `base_score` differently in the model config.
Rather than trying to parse it correctly across versions, the generator probes
the model: it evaluates a sample through all trees manually, compares against
XGBoost's `predict(output_margin=True)`, and computes the offset. This offset
is embedded as `BASE_SCORE` in the generated code.

**Thread mapping (CUDA)**
One thread per sample is the default. Each thread traverses all N trees
sequentially. This is optimal when the number of samples is large (thousands+),
which is the expected use case for GPU batch scoring. The block size is 256
threads, a standard choice for compute-bound kernels.

## CUDA Execution Model

This section describes exactly how the generated kernel maps to GPU hardware.

### Thread-to-Sample Mapping

The kernel uses a **1:1 mapping** of CUDA threads to input samples:

```
score_kernel<<<grid_size, block_size>>>(d_features, d_scores, n_samples);

where:
  block_size = 256 threads
  grid_size  = ceil(n_samples / 256)
```

Each thread computes:
```
thread i:
    f = features[i * 39 .. i * 39 + 38]   // load 39 floats for this sample
    sum = tree_0(f) + tree_1(f) + ... + tree_299(f)   // walk all 300 trees
    scores[i] = sigmoid(BASE_SCORE + sum)              // output probability
```

Threads are completely independent — no shared memory, no synchronization,
no inter-thread communication. This makes the kernel embarrassingly parallel.

### GPU Utilization and Hardware Mapping

**Do all CUDA cores get used?**

Yes, provided the batch size is large enough. Here's how it maps to hardware:

```
Batch size     Threads launched    Blocks (of 256)    GPU saturation
─────────────────────────────────────────────────────────────────────
100            100                 1                  ~1 SM active (underutilized)
1,000          1,000               4                  4 SMs active (still low)
10,000         10,000              40                 40 SMs active (moderate)
50,000         50,000              196                Full saturation on most GPUs
100,000        100,000             391                Fully saturated with overlap
1,000,000      1,000,000           3,907              Multiple waves per SM
```

Reference GPU core counts:
- NVIDIA A100: 108 SMs × 64 cores = 6,912 CUDA cores
- NVIDIA H100: 132 SMs × 128 cores = 16,896 CUDA cores
- NVIDIA RTX 4090: 128 SMs × 128 cores = 16,384 CUDA cores
- NVIDIA T4: 40 SMs × 64 cores = 2,560 CUDA cores

To fully saturate the GPU, you need enough blocks to keep all SMs busy.
Each SM can run multiple blocks concurrently (typically 2-4 depending on
register and shared memory usage). So for an A100, you want at least
~200-400 blocks = **50K-100K samples** for peak throughput.

### What Each Thread Does (Detailed)

For this model (300 trees, depth 6, 39 features), each thread executes:

1. **Compute feature pointer**: 1 multiply + 1 add (index into the flat array)
2. **Traverse 300 trees**: Each tree is 6 levels of if/else = 6 comparisons
   per tree. Total: ~1,800 float comparisons + ~1,800 branches.
3. **Sum 300 leaf values**: 300 float additions.
4. **Apply sigmoid**: 1 negation + 1 expf + 1 division.

Total per thread: ~1,800 comparisons, ~300 additions, 1 expf.
This is heavily compute-bound (very little memory access relative to
computation), which is ideal for GPU execution.

### Warp Divergence

A CUDA warp is 32 threads that execute in lockstep. When threads in the same
warp take different branches at an if/else node, both paths must be executed
(with some threads masked). This is called "warp divergence."

For tree traversal, divergence is expected — different samples will have
different feature values and take different paths through each tree. However:

- Each branch is very short (just a return or another comparison)
- The trees are only 6 levels deep, limiting the divergence penalty
- With 300 trees, the total work per thread is large enough that divergence
  overhead is a small fraction of total execution time
- In practice, many samples in a warp will share similar paths (especially
  near the root of each tree where splits are on high-importance features)

For this model, warp divergence is not a significant bottleneck.

### Memory Access Pattern

Each thread reads 39 floats from the feature array (its own sample's row).
With the row-major layout, consecutive threads in a warp access consecutive
rows, which means the reads are strided (not coalesced). However:

- Each sample is only 39 × 4 = 156 bytes, which fits in one or two cache
  lines (128 bytes each)
- The feature values are read from L1/L2 cache after the first tree
  accesses them, so the 300 subsequent tree traversals are served from cache
- The output write (one float per thread) is coalesced since threads write
  to consecutive addresses in the scores array

### Alternative: Parallelism Across Trees

The current design is **sample-parallel**: each thread handles one sample,
all trees. An alternative design is **tree-parallel**: assign a group of
threads to each sample, with each thread handling a subset of trees, then
reduce (sum) within the group.

```
Sample-parallel (current):       Tree-parallel (alternative):
  Thread 0 → sample 0, all 300     Thread 0-7 → sample 0, 37-38 trees each
  Thread 1 → sample 1, all 300     Thread 8-15 → sample 1, 37-38 trees each
  ...                               ... (requires warp-level reduction)
```

Tree-parallel would help when:
- Batch sizes are very small (< 100 samples) and you need more parallelism
- The ensemble is very large (10,000+ trees) and latency matters

Tree-parallel hurts when:
- Batch sizes are large (the common case) — it reduces the number of samples
  you can process concurrently
- The reduction step adds synchronization overhead

For typical batch scoring (1K+ samples), sample-parallel is the right choice.

### Throughput Estimates

Rough back-of-envelope for this model (300 trees, depth 6, 39 features):

- **A100 GPU**: ~2-5M samples/second (depending on batch size)
- **T4 GPU**: ~0.5-1.5M samples/second
- **CPU single-core (generated C)**: ~10-50K samples/second
- **XGBoost Python predict()**: ~5-30K samples/second

These are estimates — actual throughput depends on GPU clock speed, memory
bandwidth, and thermal conditions. The key takeaway is ~100x speedup over
CPU for large batches.

## Split-Feature Mode (Ad Ranking / Recommendation)

### The Problem

In ad ranking and recommendation, the scoring model was trained on a combined
feature vector that represents a (user, item) pair. At training time, each
row in the dataset looks like:

```
[user_feat_0, user_feat_1, ..., user_feat_U-1, item_feat_0, item_feat_1, ..., item_feat_I-1]
```

The model learned splits across **all** features — some tree nodes split on
user features, others on item features. The model is a single ensemble; it
cannot be separated into "user model" and "item model."

At serving time, the scenario is fundamentally asymmetric:
- There is **one user** (or user context) per request.
- There are **millions of candidate items** (ads, products, documents) to rank.
- Every (user, item) pair shares the same user features — only the item
  features differ.

Naively, you would construct N full feature vectors (one per item), each
containing a redundant copy of the user features, and transfer all N × F
floats to the GPU for every request. For 1M items × 39 features, that is
156 MB per request — most of which is duplicated user data.

### The Solution: Split the Feature Vector

Split-feature mode separates the feature vector into two parts at code
generation time:

```
Original model training feature vector (39 features):
┌──────────────────────────────┬──────────────────────────────────────────────┐
│  User / Context features     │  Item / Document / Ad features               │
│  f[0] .. f[12]               │  f[13] .. f[38]                              │
│  I1 I2 I3 I4 I5 I6 I7 I8    │  C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 ...  C26   │
│  I9 I10 I11 I12 I13          │                                              │
│                              │                                              │
│  13 floats (52 bytes)        │  26 floats (104 bytes) × N items             │
│  Changes per request         │  Loaded once, reused across requests         │
└──────────────────────────────┴──────────────────────────────────────────────┘
```

The `--user-features 13` flag tells the code generator: "features at positions
0 through 12 are the user/context part; features 13 through 38 are the item
part." The generated code then:

1. **Stores item features on GPU** — uploaded once via `load_items()`. This
   is your item catalog (ads, products, documents). It stays resident until
   the catalog changes.
2. **Transfers only the user vector per request** — 13 floats = 52 bytes.
   This is the user's context: demographics, session history, etc.
3. **Composes the full feature vector at runtime** — each GPU thread (or CPU
   loop iteration) concatenates the user vector with its item's features into
   a local `f[39]` array, then runs all 300 trees against it.

### Feature Ordering: What Must Match Training

The split is position-based. The generated code uses the same feature indices
that the XGBoost model was trained on. For example, if the model was trained
with feature names `[I1, I2, ..., I13, C1, C2, ..., C26]`:

```
f[0]  = I1    ─┐
f[1]  = I2     │
f[2]  = I3     │ User features: supplied per request
...            │ (must be in EXACTLY this order)
f[12] = I13   ─┘

f[13] = C1    ─┐
f[14] = C2     │
f[15] = C3     │ Item features: pre-loaded on GPU
...            │ (one row per item, row-major, same order)
f[38] = C26   ─┘
```

The tree functions reference features by index (e.g. `f[9]` for I10, `f[13]`
for C1). If you supply user features in a different order than training, the
scores will be wrong. The generated header includes comments on every tree
node showing which named feature it splits on.

### How to Prepare the Data

**User vector** — a flat array of `NUM_USER_FEATURES` floats in training order:

```c
float user[NUM_USER_FEATURES];  // = float user[13];
user[0]  = I1_value;   // e.g. page views
user[1]  = I2_value;   // e.g. session count
...
user[12] = I13_value;  // e.g. days since last visit
```

**Item features** — a flat row-major array of `n_items × NUM_ITEM_FEATURES`
floats. Each row is one item's features in training order:

```c
// Row-major: item 0 features, then item 1 features, then ...
float items[n_items * NUM_ITEM_FEATURES];  // = float items[n_items * 26];

// Item 0:
items[0 * 26 + 0]  = C1_value_item0;   // e.g. ad category
items[0 * 26 + 1]  = C2_value_item0;   // e.g. advertiser ID
...
items[0 * 26 + 25] = C26_value_item0;

// Item 1:
items[1 * 26 + 0]  = C1_value_item1;
...
```

Note: the item array contains **only item features** (26 per item), not the
full 39. You do NOT include user features in the item rows. The runtime
assembles the full vector `f[0..38]` by concatenating user + item.

**From Python / NumPy:**

```python
import numpy as np

# Suppose you have the same DataFrame used for training:
# X has columns [I1, I2, ..., I13, C1, C2, ..., C26]

# Extract user features for one user (positions 0..12)
user = X.iloc[0, :13].values.astype(np.float32)       # shape: (13,)
user.tofile("user.bin")

# Extract item features for all items (positions 13..38)
# Items is an N×26 matrix — only the item columns
items = item_catalog_df[["C1","C2",...,"C26"]].values.astype(np.float32)  # (N, 26)
items.tofile("items.bin")
```

### Invoking the Scorer

**CUDA (GPU) — two-phase API:**

```c
#include "scoring_split_core.cuh"

// Phase 1: Load item catalog (once at startup, or when catalog changes)
ScoringContext* ctx = scoring_context_create(n_items);
load_items(ctx, item_features, n_items);  // H→D: n_items × 26 × 4 bytes

// Phase 2: Score one user against all items (per request)
float user[NUM_USER_FEATURES];   // 13 floats — fill from user context
float scores[n_items];            // output: one score per item
score_user(ctx, user, scores);    // H→D: 52 bytes, D→H: n_items × 4 bytes

// Score another user — items stay on GPU, no re-upload
score_user(ctx, another_user, scores);

// Top-K: get just the best K items without transferring all scores
int    topk_idx[100];
float  topk_scores[100];
score_user_topk(ctx, user, 100, topk_idx, topk_scores);
// topk_idx[0] = best item index, topk_scores[0] = highest probability

// Cleanup
scoring_context_destroy(ctx);
```

**CPU — single-call API:**

```c
#include "scoring_split_core.h"

float user[NUM_USER_FEATURES];                // 13 floats
float items[n_items * NUM_ITEM_FEATURES];     // n × 26 floats (row-major)
float scores[n_items];                         // output

score_user_cpu(user, items, scores, n_items);
// scores[i] = P(click) for user × item[i]

// Top-K on CPU: score then qsort
ScoredItem ranked[n_items];
for (int i = 0; i < n_items; i++) {
    ranked[i].score = scores[i];
    ranked[i].index = i;
}
qsort(ranked, n_items, sizeof(ScoredItem), scored_item_cmp_desc);
// ranked[0].index = best item, ranked[0].score = highest score
```

**CLI (standalone binary):**

```bash
# user.bin = float32[13], items.bin = float32[N × 26]
../build/scoring_split_cpu user.bin items.bin output.bin

# With top-K:
../build/scoring_split_cpu user.bin items.bin output.bin -k 100
```

### Data Transfer Comparison

For 1M items with 13 user features and 26 item features:

```
Standard mode (per request):
  Host → Device: 1M × 39 × 4 bytes = 156 MB   (full feature matrix)
  Device → Host: 1M × 4 bytes = 4 MB           (scores)

Split-feature mode (per request):
  Host → Device: 13 × 4 bytes = 52 bytes        (user vector only!)
  Device → Host: 1M × 4 bytes = 4 MB            (scores)
  Item features: 0 bytes                         (already on GPU)
```

This is a **~3,000,000x reduction** in host→device transfer per request.
With top-K, the D→H transfer also drops dramatically (e.g. 3.2 MB for K=100
across 4096 blocks instead of 4 MB for all 1M scores).

### How the Split Kernel Works

The tree functions are identical to standard mode — they take `const float* f`
where `f` is the full 39-element feature vector. The difference is that the
kernel (or CPU loop) **assembles** `f` at runtime from the two separate arrays:

```
__global__ void score_kernel(d_user, d_items, scores, n_items) {
    int idx = threadIdx + blockIdx * blockDim;
    float f[39];                      // local array (registers/stack)
    f[0..12] = d_user[0..12];         // user part — same for all threads
    f[13..38] = d_items[idx][0..25];  // item part — unique per thread
    score = sigmoid(BASE_SCORE + score_all_trees(f));
}
```

On GPU, the user features are broadcast to all threads. Because every thread
in a warp reads the same user addresses, the hardware serves this from L1
cache after the first read. The item features are per-thread and read from
global memory (cached in L2).

On CPU, `score_user_cpu` does the same in a loop:
```c
for (int idx = 0; idx < n_items; idx++) {
    float f[39];
    memcpy(f, user_features, 13 * sizeof(float));
    memcpy(f + 13, item_features + idx * 26, 26 * sizeof(float));
    scores[idx] = sigmoid(BASE_SCORE + score_all_trees(f));
}
```

### Applying This to Your Own Model

The split is determined by the `--user-features N` flag at code generation
time. The value of N tells the generator:

- Features `0..N-1` = user/context (supplied per request)
- Features `N..end` = item/document/ad (pre-loaded)

This means your training data must be organized so that user features come
first and item features come last. If your training data has features
interleaved (e.g. user, item, user, item), you need to reorder columns
before training, or adjust the feature name mapping so the generator can
split correctly.

The generated constants tell you exactly what the split looks like:
```c
#define NUM_FEATURES       39   // total (must match training)
#define NUM_USER_FEATURES  13   // first 13 positions
#define NUM_ITEM_FEATURES  26   // remaining 26 positions
```

## Top-K Ranking (Per-Block Shared Memory)

For ad ranking, you typically don't need all N scores — just the **top K**
(e.g. K=100 out of 10M items). The top-K feature avoids transferring all N
scores back to the host by performing selection directly on the GPU.

### Algorithm: Two-Phase Top-K

**Phase 1 — Per-Block Top-K (GPU kernel: `score_topk_kernel`)**

Each block processes a chunk of items (`n_items / grid_size`). Within each
block, items are scored in batches of `TOPK_BLOCK_SIZE` (256) threads in
parallel. After each batch, thread 0 merges the batch results into a sorted
top-K list maintained in shared memory using binary-search insertion.

No global atomics are used — each block operates independently on its chunk.

```
Block 0: items[0..2499]      → local top-100 in shared memory → write to global
Block 1: items[2500..4999]   → local top-100 in shared memory → write to global
...                            ...
Block 4095: items[...end]    → local top-100 in shared memory → write to global
```

**Phase 2 — Merge Block Results (host: `qsort`)**

The host collects `grid_size × K` candidates (e.g. 4096 × 100 = 409,600) and
runs `qsort` to extract the final global top-K. This is fast because only a
small fraction of the total items are transferred and sorted.

### Shared Memory Layout

```
Dynamic shared memory per block:
  float  s_topk_scores [K]              — current block top-K (sorted descending)
  int    s_topk_indices[K]              — corresponding item indices
  float  s_batch_scores [TOPK_BLOCK_SIZE] — batch buffer for parallel scoring
  int    s_batch_indices[TOPK_BLOCK_SIZE] — batch buffer indices

Total: (K + 256) × 8 bytes
  K=100:  2,848 bytes
  K=1024: 10,240 bytes
```

### Performance Characteristics

The insertion sort on thread 0 adds negligible overhead relative to the scoring
phase. Each tree traversal involves ~1,800 comparisons per item; the top-K
insertion is at most `O(K)` = 100 comparisons per candidate. Since only a
small fraction of candidates beat the current minimum, the amortized insertion
cost per item is much less than 1% of the scoring cost.

```
Example: 10M items, K=100, block_size=256, grid_size=4096

Phase 1 (GPU):   10M items scored across 4096 blocks
                 Each block: ~2,500 items → 10 batches of 256
                 Per-block output: 100 candidates (sorted descending)

Phase 2 (host):  409,600 candidates → qsort → final top-100
                 D→H transfer: 409K × 8 bytes = 3.2 MB (vs 40 MB for all 10M scores)
```

### Data Transfer Savings

For 10M items with K=100:

```
score_user():       D→H = 10M × 4 bytes = 40 MB
score_user_topk():  D→H = 4096 × 100 × 8 bytes = 3.2 MB  (12.5× reduction)
```

Combined with split-feature mode (52 bytes H→D for user features), the total
PCIe transfer per request drops to ~3.2 MB.

### API Usage

```c
// Create context (allocates scoring + top-K buffers)
ScoringContext* ctx = scoring_context_create(max_items);
load_items(ctx, item_features, n_items);

// Top-K mode: returns K best items (no full score array needed)
int    topk_indices[100];
float  topk_scores[100];
score_user_topk(ctx, user_features, 100, topk_indices, topk_scores);
// topk_indices[0] = best item index, topk_scores[0] = highest score

// Full scoring mode still available:
float all_scores[n_items];
score_user(ctx, user_features, all_scores);

scoring_context_destroy(ctx);
```

### CLI Usage

```bash
# Generate split-feature code (includes both score_user and score_user_topk)
PYTHONPATH=python python3 -m cuda_codegen generate \
    --model models/model.ubj \
    --metadata models/training_result.json \
    --output generated/scoring_split.cu \
    --user-features 13

# Run compiled binary with top-K
./build/scoring_split user.bin items.bin output.bin -k 100

# Verify top-K correctness (CPU path)
PYTHONPATH=python python3 -m cuda_codegen verify \
    --model models/model.ubj \
    --metadata models/training_result.json \
    --user-features 13 \
    --topk 100 \
    --n-samples 5000
```

## Library Mode (Recommended)

Library mode generates the scoring code as a self-contained directory with
separate header-only core, standalone drivers, benchmarks, and a Makefile.
This makes it easy to include the generated scoring function in your own
runtime.

### Generating the Library

```bash
# Generate the library (13 user features for the Criteo model)
./scripts/generate_split_library.sh 13

# Or via CLI directly
PYTHONPATH=python python3 -m cuda_codegen generate \
    --model models/model.ubj \
    --metadata models/training_result.json \
    --output generated \
    --user-features 13 \
    --library
```

This produces:

```
generated/
├── scoring_split_core.cuh   # CUDA header-only library (all trees, kernels, host API)
├── scoring_split_core.h     # CPU header-only library
├── scoring_split_main.cu    # CUDA standalone main driver
├── scoring_split_main.c     # CPU standalone main driver
├── scoring_split_bench.cu   # CUDA benchmark driver
├── scoring_split_bench.c    # CPU benchmark driver
└── Makefile                 # Build targets: cpu, cuda, bench_cpu, bench_cuda, all, clean
```

### Building

```bash
cd generated

# Build CPU main + benchmark
make cpu

# Build CUDA main + benchmark (requires nvcc)
make cuda

# Build everything available
make all

# Override compiler or flags
make CC=gcc-13 CFLAGS="-O3 -march=native" cpu
```

### Using the Core Headers

To integrate scoring into your own runtime, just include the core header:

```c
// CPU scoring
#include "scoring_split_core.h"

float user[NUM_USER_FEATURES];
float items[n_items * NUM_ITEM_FEATURES];
float scores[n_items];
score_user_cpu(user, items, scores, n_items);
```

```c
// CUDA scoring
#include "scoring_split_core.cuh"

ScoringContext* ctx = scoring_context_create(n_items);
load_items(ctx, items, n_items);
score_user(ctx, user, scores);
// or: score_user_topk(ctx, user, k, topk_idx, topk_scores);
scoring_context_destroy(ctx);
```

### Benchmarks

The benchmark drivers measure scoring throughput and top-K overhead with
configurable parameters:

```bash
# Run CPU benchmark (default: 100K items, K=100, 5 iterations)
make bench_cpu

# Customize: 1M items, top-500, 20 iterations
make bench_cpu BENCH_ARGS="-n 1000000 -k 500 -i 20"

# Run CUDA benchmark
make bench_cuda BENCH_ARGS="-n 10000000 -k 100 -i 50"

# Or run the binary directly
../build/scoring_split_bench_cpu -n 500000 -k 200 -i 10 -w 3
```

**Benchmark parameters:**

- `-n <items>` — Number of item vectors to score (default: 100K CPU, 1M CUDA)
- `-k <topK>` — Top-K selection size (default: 100)
- `-i <iters>` — Number of timed iterations to average (default: 5 CPU, 20 CUDA)
- `-w <warmup>` — Warmup iterations before timing (default: 2 CPU, 5 CUDA)
- `-s <seed>` — RNG seed for reproducible random data (default: 42)

**Example: Scoring 2M ads on CPU**

```bash
../build/scoring_split_bench_cpu -n 2000000 -k 100 -w 5 -i 10
```

```
=== CPU Split-Feature Scoring Benchmark ===
  Items:         2000000
  Top-K:         100
  Trees:         300
  Warmup:        5
  Iterations:    10

Generated 198.4 MB of random item features

--- Results ---
  Full scoring:          594.68 ms  ( 3363.15 K items/sec)
  Score + top-100 :      599.13 ms  ( 3338.17 K items/sec)
  Top-K overhead:          4.45 ms  (0.7% of scoring)

--- Top-K Scaling (score + qsort) ---
  K=10         598.75 ms  ( 3340.27 K items/sec)
  K=50         598.45 ms  ( 3341.98 K items/sec)
  K=100        590.47 ms  ( 3387.16 K items/sec)
  K=500        602.98 ms  ( 3316.84 K items/sec)
  K=1000       596.36 ms  ( 3353.70 K items/sec)
```

Single-core CPU throughput: ~3.4M items/sec (300 trees, depth 6). Top-K
selection via qsort adds <1% overhead regardless of K.

**Benchmark output includes:**

- Full scoring throughput (K items/sec for CPU, M items/sec for CUDA)
- Score + top-K throughput and overhead percentage
- Top-K scaling across K={10, 50, 100, 500, 1000}
- Preview of top-ranked items
- CUDA benchmark also reports GPU info, H→D load time, and D→H transfer savings

## Quick Start

### Prerequisites

- Python 3.10+ with `xgboost`, `numpy` installed
- A trained XGBoost model (`.ubj` format)
- `gcc` or `clang` (for CPU builds and verification)
- `nvcc` / CUDA Toolkit (for CUDA builds — optional for development)

### 1. Generate code

```bash
# Generate both CUDA and CPU versions from the default model
./scripts/generate_cuda.sh

# Or use the Python CLI directly
PYTHONPATH=python python -m cuda_codegen generate \
    --model models/model.ubj \
    --metadata models/training_result.json \
    --output generated/scoring_kernel.cu

# CPU-only version (for testing without CUDA)
PYTHONPATH=python python -m cuda_codegen generate \
    --model models/model.ubj \
    --metadata models/training_result.json \
    --output generated/scoring_kernel.c \
    --cpu
```

### 2. Verify correctness

This compiles the generated C code, runs it on random test data, and compares
against XGBoost Python predictions:

```bash
./scripts/verify_scoring.sh

# Or with custom parameters
PYTHONPATH=python python -m cuda_codegen verify \
    --model models/model.ubj \
    --metadata models/training_result.json \
    --n-samples 5000 \
    --tolerance 1e-5
```

### 3. Build

```bash
# CPU only (works on any machine)
./scripts/build_cuda.sh cpu

# CUDA (requires nvcc)
./scripts/build_cuda.sh cuda

# Both
./scripts/build_cuda.sh both
```

### 4. Run

The compiled binary reads feature vectors from a binary float32 file and
outputs scores:

```bash
# Print scores to stdout
./build/scoring_cpu features.bin

# Write scores to binary file
./build/scoring_cpu features.bin scores.bin

# On GPU
./build/scoring_cuda features.bin scores.bin
```

**Binary file format:**
- Input: `float32[N × NUM_FEATURES]`, row-major (C order)
- Output: `float32[N]`, one probability per sample

You can create test input from Python:
```python
import numpy as np
X = np.random.randn(10000, 39).astype(np.float32)
X.tofile("features.bin")
```

### 5. Use as a library (CUDA)

The generated CUDA code exposes a C API for integration into larger systems:

```c
#include "scoring_kernel.cu"  // or compile separately and link

// Create context (allocates GPU memory)
ScoringContext* ctx = scoring_context_create(max_batch_size);

// Score a batch
float features[N * NUM_FEATURES];  // fill with data
float scores[N];
score_batch(ctx, features, scores, N);

// Reuse context for multiple batches
score_batch(ctx, next_features, next_scores, M);

// Cleanup
scoring_context_destroy(ctx);
```

## Python API

```python
from cuda_codegen import CudaCodeGenerator

# Load model and create generator
gen = CudaCodeGenerator(
    "models/model.ubj",
    metadata_path="models/training_result.json",  # optional, for feature names
)

# Inspect model info
print(gen.model_info.num_trees)      # 300
print(gen.model_info.num_features)   # 39
print(gen.model_info.objective)      # binary:logistic
print(gen.model_info.base_score)     # calibrated offset

# Generate code
gen.generate_cuda("output/kernel.cu")
gen.generate_cpu("output/kernel.c")
```

## File Layout

```
python/cuda_codegen/
├── __init__.py        # Package exports
├── __main__.py        # python -m cuda_codegen entry point
├── generator.py       # Core: XGBoost model → CUDA/C source
├── verify.py          # Correctness verification (Python vs compiled C)
└── cli.py             # CLI (generate / verify commands)

scripts/
├── generate_cuda.sh           # Generate .cu and .c from model
├── generate_split_cuda.sh     # Generate split-feature .cu/.c
├── generate_split_library.sh  # Generate library (core + drivers + bench + Makefile)
├── build_cuda.sh              # Compile generated code (cpu/cuda/both)
└── verify_scoring.sh          # End-to-end correctness check

generated/                     # (git-ignored) generated source files
├── scoring_split_core.cuh     # CUDA header-only library (library mode)
├── scoring_split_core.h       # CPU header-only library (library mode)
├── scoring_split_main.cu/.c   # Standalone main drivers (library mode)
├── scoring_split_bench.cu/.c  # Benchmark drivers (library mode)
├── Makefile                   # Build targets (library mode)
├── scoring_kernel.cu          # CUDA version (single-file mode)
└── scoring_kernel.c           # CPU version (single-file mode)

build/                         # (git-ignored) compiled binaries
├── scoring_split_cpu          # CPU main (library mode)
├── scoring_split_cuda         # CUDA main (library mode)
├── scoring_split_bench_cpu    # CPU benchmark (library mode)
├── scoring_split_bench_cuda   # CUDA benchmark (library mode)
├── scoring_cuda               # CUDA binary (single-file mode)
└── scoring_cpu                # CPU binary (single-file mode)
```

## Supported Objectives

The generator maps XGBoost objectives to output transformations:

- `binary:logistic` → sigmoid: `1 / (1 + exp(-raw))`
- `binary:logitraw` → identity: `raw`
- `reg:squarederror` → identity: `raw`
- `rank:pairwise` → sigmoid
- `rank:ndcg` → sigmoid

Additional objectives can be added by extending `_CUDA_TRANSFORMS` and
`_CPU_TRANSFORMS` in `generator.py`.

## Performance Considerations

- **GPU vs CPU:** For batch sizes of 10K+ samples, the CUDA version will be
  significantly faster due to massive parallelism (one thread per sample).
  For small batches (<100 samples), the CPU version may be faster due to
  GPU launch overhead and memory transfer costs.

- **Compilation time:** The generated code for a 300-tree, depth-6 model is
  ~75K lines of C. `gcc -O2` compiles this in ~5-10 seconds. `nvcc -O2` takes
  ~15-30 seconds. Use `-O1` or `-O0` for faster iteration during development.

- **Instruction cache:** For models with >1000 trees, the inlined if/else
  approach may cause instruction cache pressure. Consider splitting into
  multiple translation units or switching to a data-driven tree traversal.

- **Memory transfer:** For high-throughput scenarios, use CUDA streams to
  overlap host-to-device transfer with kernel execution. The `ScoringContext`
  can be extended with double-buffering for this purpose.
