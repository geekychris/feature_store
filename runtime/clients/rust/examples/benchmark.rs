use scorched_client::ScoringClient;
use std::time::Instant;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let addr = std::env::var("SCORCHED_ADDR").unwrap_or_else(|_| "http://localhost:50051".into());
    let n_items: i32 = std::env::var("BENCH_ITEMS")
        .unwrap_or_else(|_| "100000".into())
        .parse()?;
    let n_requests: usize = std::env::var("BENCH_REQUESTS")
        .unwrap_or_else(|_| "1000".into())
        .parse()?;
    let top_k: i32 = std::env::var("BENCH_TOPK")
        .unwrap_or_else(|_| "100".into())
        .parse()?;

    println!("=== Scorched Rust Client Benchmark ===");
    println!("  Server:   {}", addr);
    println!("  Items:    {}", n_items);
    println!("  Requests: {}", n_requests);
    println!("  Top-K:    {}", top_k);

    let mut client = ScoringClient::connect(&addr).await?;

    // Load dataset
    let num_features = 26i32;
    let mut data = Vec::with_capacity((n_items * num_features) as usize);
    for i in 0..n_items {
        for f in 0..num_features {
            data.push((i as f32 * 0.001) + (f as f32 * 0.01));
        }
    }

    let load_resp = client
        .load_dataset("bench", data, n_items, num_features, true)
        .await?;
    println!(
        "\nDataset loaded: {} items in {}µs",
        load_resp.num_items, load_resp.load_duration_us
    );

    // Warmup
    let user: Vec<f32> = (0..13).map(|i| i as f32).collect();
    for _ in 0..10 {
        client.score_topk("bench", user.clone(), top_k).await?;
    }

    // Benchmark
    let mut latencies = Vec::with_capacity(n_requests);
    let total_start = Instant::now();

    for _ in 0..n_requests {
        let start = Instant::now();
        let _resp = client.score_topk("bench", user.clone(), top_k).await?;
        latencies.push(start.elapsed().as_micros() as u64);
    }

    let total_elapsed = total_start.elapsed();
    latencies.sort();

    let p50 = latencies[n_requests / 2];
    let p95 = latencies[(n_requests as f64 * 0.95) as usize];
    let p99 = latencies[(n_requests as f64 * 0.99) as usize];
    let avg = latencies.iter().sum::<u64>() / n_requests as u64;
    let qps = n_requests as f64 / total_elapsed.as_secs_f64();

    println!("\n--- Results ---");
    println!("  Total time: {:.2}s", total_elapsed.as_secs_f64());
    println!("  QPS:        {:.1}", qps);
    println!("  Avg:        {}µs", avg);
    println!("  p50:        {}µs", p50);
    println!("  p95:        {}µs", p95);
    println!("  p99:        {}µs", p99);
    println!(
        "  Throughput: {:.2}M items/sec",
        (n_requests as f64 * n_items as f64) / total_elapsed.as_secs_f64() / 1_000_000.0
    );

    Ok(())
}
