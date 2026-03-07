// Example: connect to scorched, load a dataset, score, and benchmark.
//
// Usage:
//
//	go run ./examples/main.go --addr localhost:9090
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"time"

	scorched "github.com/platform/scorched"
)

func main() {
	addr := flag.String("addr", "localhost:9090", "scorched gRPC address")
	numItems := flag.Int("items", 5000, "number of synthetic items to load")
	topK := flag.Int("k", 10, "top-K to return")
	benchN := flag.Int("bench", 1000, "benchmark iterations")
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client, err := scorched.Connect(*addr)
	if err != nil {
		log.Fatalf("connect: %v", err)
	}
	defer client.Close()

	// Health check
	health, err := client.Healthz(ctx)
	if err != nil {
		log.Fatalf("healthz: %v", err)
	}
	fmt.Printf("Health: %v  CUDA: %v  Datasets: %d\n",
		health.Status, health.CudaAvailable, health.DatasetsLoaded)

	// Build synthetic dataset: numItems x 26 item features
	const numFeatures int32 = 26
	items := *numItems
	data := make([]float32, items*int(numFeatures))
	for i := range data {
		data[i] = rand.Float32()
	}

	// Load dataset
	fmt.Printf("Loading %d items (%d features each)...\n", items, numFeatures)
	loadResp, err := client.LoadDataset(ctx, "bench_dataset", data, int32(items), numFeatures, true)
	if err != nil {
		log.Fatalf("load: %v", err)
	}
	fmt.Printf("Loaded: %s (%d items, %d bytes, %d µs)\n",
		loadResp.DatasetName, loadResp.NumItems, loadResp.MemoryBytes, loadResp.LoadDurationUs)

	// Score once
	userVec := make([]float32, 13) // 13 user features
	for i := range userVec {
		userVec[i] = rand.Float32()
	}

	scoreResp, err := client.ScoreTopK(ctx, "bench_dataset", userVec, int32(*topK))
	if err != nil {
		log.Fatalf("score: %v", err)
	}
	fmt.Printf("\nTop-%d results (%d items scored in %d µs, backend=%s):\n",
		*topK, scoreResp.ItemsScored, scoreResp.LatencyUs, scoreResp.Backend)
	for _, c := range scoreResp.Candidates {
		fmt.Printf("  [%5d] %.6f\n", c.Index, c.Score)
	}

	// Benchmark
	fmt.Printf("\nBenchmarking %d iterations...\n", *benchN)
	benchCtx := context.Background()
	result, err := client.Benchmark(benchCtx, "bench_dataset", userVec, int32(*topK), *benchN)
	if err != nil {
		log.Fatalf("benchmark: %v", err)
	}
	fmt.Printf("Results (%d requests):\n", result.Requests)
	fmt.Printf("  Total:  %v\n", result.TotalTime)
	fmt.Printf("  P50:    %v\n", result.P50)
	fmt.Printf("  P95:    %v\n", result.P95)
	fmt.Printf("  P99:    %v\n", result.P99)
	fmt.Printf("  QPS:    %.1f\n", result.QPS)
}
