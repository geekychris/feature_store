fn main() {
    tonic_build::configure()
        .build_server(false)
        .build_client(true)
        .compile_protos(&["../../proto/scoring.proto"], &["../../proto/"])
        .expect("Failed to compile scoring.proto");
}
