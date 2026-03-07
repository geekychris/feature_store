/*
 * FFI shim — includes the auto-generated header-only scoring library
 * so that cc can compile it into a static library for Rust FFI.
 *
 * DO NOT add scoring logic here. The generated code is the source of truth.
 */

#include "scoring_split_core.h"

/*
 * Explicit non-inline wrappers so that Rust can link to them.
 * The header uses `static inline` for all tree functions, so we need
 * these thin wrappers with external linkage.
 */

void scorched_score_user_cpu(
    const float* user_features,
    const float* item_features,
    float* scores,
    int n_items
) {
    score_user_cpu(user_features, item_features, scores, n_items);
}

/* Top-K helper: score + sort, return top-K indices and scores */
int scorched_score_topk_cpu(
    const float* user_features,
    const float* item_features,
    int n_items,
    int k,
    int* out_indices,
    float* out_scores
) {
    float* scores = (float*)malloc((size_t)n_items * sizeof(float));
    if (!scores) return -1;

    score_user_cpu(user_features, item_features, scores, n_items);

    ScoredItem* ranked = (ScoredItem*)malloc((size_t)n_items * sizeof(ScoredItem));
    if (!ranked) { free(scores); return -1; }

    for (int i = 0; i < n_items; i++) {
        ranked[i].score = scores[i];
        ranked[i].index = i;
    }
    qsort(ranked, n_items, sizeof(ScoredItem), scored_item_cmp_desc);

    int result_k = k < n_items ? k : n_items;
    for (int i = 0; i < result_k; i++) {
        out_indices[i] = ranked[i].index;
        out_scores[i]  = ranked[i].score;
    }
    /* Fill remaining slots with sentinel */
    for (int i = result_k; i < k; i++) {
        out_indices[i] = -1;
        out_scores[i]  = -1e30f;
    }

    free(ranked);
    free(scores);
    return result_k;
}

/* Query constants so Rust can discover them at runtime */
int scorched_num_features(void)      { return NUM_FEATURES; }
int scorched_num_user_features(void) { return NUM_USER_FEATURES; }
int scorched_num_item_features(void) { return NUM_ITEM_FEATURES; }
int scorched_num_trees(void)         { return NUM_TREES; }
float scorched_base_score(void)      { return BASE_SCORE; }
