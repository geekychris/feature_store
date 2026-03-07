PROTO_DIR = ../../proto
OUT_DIR   = .

.PHONY: proto clean

proto:
	protoc \
		--proto_path=$(PROTO_DIR) \
		--go_out=$(OUT_DIR) --go_opt=paths=source_relative \
		--go-grpc_out=$(OUT_DIR) --go-grpc_opt=paths=source_relative \
		$(PROTO_DIR)/scoring.proto

clean:
	rm -f scoring.pb.go scoring_grpc.pb.go
