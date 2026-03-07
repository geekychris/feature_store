use scorched_client::ScoringClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut client = ScoringClient::connect("http://localhost:50051").await?;

    // Check health
    let health = client.healthz().await?;
    println!("Health: {:?}", health);

    // Generate synthetic item data: 10,000 items × 26 features
    let num_items = 10_000i32;
    let num_features = 26i32;
    let mut candidate_matrix = Vec::with_capacity((num_items * num_features) as usize);
    for i in 0..num_items {
        for f in 0..num_features {
            candidate_matrix.push((i as f32 * 0.01) + (f as f32 * 0.1));
        }
    }

    // Load dataset
    let load_resp = client
        .load_dataset("demo_ads", candidate_matrix, num_items, num_features, true)
        .await?;
    println!(
        "Loaded dataset '{}': {} items, {} bytes, {}µs",
        load_resp.dataset_name, load_resp.num_items, load_resp.memory_bytes, load_resp.load_duration_us
    );

    // Score with a user vector (13 user features)
    let user_vector: Vec<f32> = (0..13).map(|i| i as f32 * 1.5).collect();
    let score_resp = client.score_topk("demo_ads", user_vector, 10).await?;
    println!(
        "\nTop-10 results ({} items scored in {}µs, backend={}):",
        score_resp.items_scored, score_resp.latency_us, score_resp.backend
    );
    for c in &score_resp.candidates {
        println!("  item[{}] = {:.8}", c.index, c.score);
    }

    // Show datasets
    let datasets = client.list_datasets().await?;
    println!("\nLoaded datasets:");
    for ds in &datasets.datasets {
        println!(
            "  {} — {} items, {} bytes, {} requests",
            ds.name, ds.num_items, ds.memory_bytes, ds.score_request_count
        );
    }

    // Stats
    let stats = client.get_stats().await?;
    println!(
        "\nStats: {} requests, {} items scored, uptime={}s",
        stats.total_score_requests, stats.total_items_scored, stats.uptime_seconds
    );

    Ok(())
}
