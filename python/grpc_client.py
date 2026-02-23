"""
Python gRPC client for the Feature Store.

Usage:
    # Generate Python protobuf stubs first:
    # python -m grpc_tools.protoc -I../src/main/proto --python_out=. --grpc_python_out=. feature_store.proto

    from grpc_client import FeatureStoreClient

    client = FeatureStoreClient("localhost:9090")
    # Write features
    client.put_feature_vector("fraud_scoring_v3", 3, "merchant", "m_001", [0.008, 45230.0, 423])
    # Read features
    vectors = client.get_online_features("fraud_scoring_v3", 3, ["m_001", "m_002"])
"""

import grpc
import time
from typing import List, Optional, Dict, Any

# These will be generated from the .proto file
import feature_store_pb2 as pb2
import feature_store_pb2_grpc as pb2_grpc


class FeatureStoreClient:
    """Python client for the Feature Store gRPC service."""

    def __init__(self, target: str = "localhost:9090"):
        self.channel = grpc.insecure_channel(target)
        self.stub = pb2_grpc.FeatureStoreServiceStub(self.channel)

    def close(self):
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # --- Online Serving ---

    def get_online_features(
        self,
        view_name: str,
        view_version: int,
        entity_ids: List[str],
        include_metadata: bool = False,
    ) -> Dict[str, Any]:
        """Fetch feature vectors for a batch of entities."""
        request = pb2.GetFeaturesRequest(
            view_name=view_name,
            view_version=view_version,
            entity_ids=entity_ids,
            include_metadata=include_metadata,
            request_id=f"py-{int(time.time()*1000)}",
        )
        response = self.stub.GetOnlineFeatures(request)

        return {
            "vectors": [
                {
                    "entity_id": v.entity_id,
                    "values": list(v.values),
                    "schema_hash": v.schema_hash,
                    "served_at_ms": v.served_at_ms,
                    "is_default_mask": list(v.is_default_mask),
                }
                for v in response.vectors
            ],
            "latency_us": response.latency_us,
            "warnings": list(response.warnings),
        }

    # --- Write Path ---

    def put_feature_vector(
        self,
        view_name: str,
        view_version: int,
        entity_type: str,
        entity_id: str,
        values: List[float],
        schema_hash: int = 0,
    ) -> bool:
        """Write a pre-materialized feature vector."""
        vector = pb2.FeatureVector(
            view_name=view_name,
            view_version=view_version,
            entity_type=entity_type,
            entity_id=entity_id,
            values=values,
            schema_hash=schema_hash,
            served_at_ms=int(time.time() * 1000),
        )
        request = pb2.PutFeatureVectorRequest(vector=vector)
        response = self.stub.PutFeatureVector(request)
        return response.success

    def put_feature_vector_batch(
        self,
        vectors: List[Dict[str, Any]],
    ) -> int:
        """Write a batch of feature vectors."""
        proto_vectors = []
        for v in vectors:
            proto_vectors.append(pb2.FeatureVector(
                view_name=v["view_name"],
                view_version=v["view_version"],
                entity_type=v["entity_type"],
                entity_id=v["entity_id"],
                values=v["values"],
                schema_hash=v.get("schema_hash", 0),
                served_at_ms=int(time.time() * 1000),
            ))
        request = pb2.PutFeatureVectorBatchRequest(vectors=proto_vectors)
        response = self.stub.PutFeatureVectorBatch(request)
        return response.vectors_written

    def put_scalar_features(
        self,
        entity_type: str,
        entity_id: str,
        features: Dict[str, float],
    ) -> int:
        """Write individual scalar features."""
        proto_features = []
        for name, value in features.items():
            proto_features.append(pb2.Feature(
                name=name,
                value=pb2.FeatureValue(float64_val=value),
                event_time_ms=int(time.time() * 1000),
            ))
        request = pb2.PutScalarFeaturesRequest(
            entity_type=entity_type,
            entity_id=entity_id,
            features=proto_features,
        )
        response = self.stub.PutScalarFeatures(request)
        return response.features_written

    # --- Schema ---

    def get_view_schema(self, view_name: str, version: int = 0) -> Dict[str, Any]:
        """Get the schema for a feature view."""
        request = pb2.GetViewSchemaRequest(view_name=view_name, version=version)
        response = self.stub.GetViewSchema(request)
        schema = response.schema
        return {
            "view_name": schema.view_name,
            "version": schema.version,
            "feature_names": list(schema.feature_names),
            "feature_dtypes": list(schema.feature_dtypes),
            "schema_hash": schema.schema_hash,
        }
