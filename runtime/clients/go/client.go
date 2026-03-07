// Package scorched provides a Go client for the scorched scoring runtime.
//
// Proto generation:
//
//	protoc --go_out=. --go-grpc_out=. --proto_path=../../proto scoring.proto
package scorched

import (
	"context"
	"fmt"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Client wraps the gRPC stubs for the scorched runtime.
type Client struct {
	conn    *grpc.ClientConn
	Score   ScoreServiceClient
	Dataset DatasetServiceClient
	Stats   StatsServiceClient
}

// Connect creates a new client connected to the given address.
func Connect(addr string) (*Client, error) {
	conn, err := grpc.NewClient(addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(64*1024*1024)),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to %s: %w", addr, err)
	}
	return &Client{
		conn:    conn,
		Score:   NewScoreServiceClient(conn),
		Dataset: NewDatasetServiceClient(conn),
		Stats:   NewStatsServiceClient(conn),
	}, nil
}

// Close closes the underlying gRPC connection.
func (c *Client) Close() error {
	return c.conn.Close()
}

// ScoreTopK scores a user against a named dataset and returns top-K candidates.
func (c *Client) ScoreTopK(ctx context.Context, dataset string, userVector []float32, topK int32) (*ScoreResponse, error) {
	return c.Score.ScoreTopK(ctx, &ScoreRequest{
		DatasetName: dataset,
		UserVector:  userVector,
		TopK:        topK,
	})
}

// LoadDataset loads a CandidateMatrix dataset into the runtime.
func (c *Client) LoadDataset(ctx context.Context, name string, data []float32, numItems, numFeatures int32, replace bool) (*LoadDatasetResponse, error) {
	return c.Dataset.LoadDataset(ctx, &LoadDatasetRequest{
		Name:             name,
		CandidateMatrix:  data,
		NumItems:         numItems,
		NumItemFeatures:  numFeatures,
		ReplaceIfExists:  replace,
	})
}

// Healthz performs a health check.
func (c *Client) Healthz(ctx context.Context) (*HealthResponse, error) {
	return c.Stats.GetHealthz(ctx, &GetHealthzRequest{})
}

// Benchmark runs n scoring requests and returns latency stats.
func (c *Client) Benchmark(ctx context.Context, dataset string, userVector []float32, topK int32, n int) (*BenchmarkResult, error) {
	// Warmup
	for i := 0; i < 10; i++ {
		if _, err := c.ScoreTopK(ctx, dataset, userVector, topK); err != nil {
			return nil, err
		}
	}

	latencies := make([]time.Duration, n)
	start := time.Now()
	for i := 0; i < n; i++ {
		t := time.Now()
		if _, err := c.ScoreTopK(ctx, dataset, userVector, topK); err != nil {
			return nil, err
		}
		latencies[i] = time.Since(t)
	}
	total := time.Since(start)

	// Sort for percentiles
	sortDurations(latencies)

	return &BenchmarkResult{
		Requests:   n,
		TotalTime:  total,
		P50:        latencies[n/2],
		P95:        latencies[int(float64(n)*0.95)],
		P99:        latencies[int(float64(n)*0.99)],
		QPS:        float64(n) / total.Seconds(),
	}, nil
}

// BenchmarkResult holds performance statistics.
type BenchmarkResult struct {
	Requests  int
	TotalTime time.Duration
	P50       time.Duration
	P95       time.Duration
	P99       time.Duration
	QPS       float64
}

func sortDurations(d []time.Duration) {
	for i := 1; i < len(d); i++ {
		key := d[i]
		j := i - 1
		for j >= 0 && d[j] > key {
			d[j+1] = d[j]
			j--
		}
		d[j+1] = key
	}
}
