package com.platform.featurestore.grpc;

import com.platform.featurestore.proto.*;
import com.platform.featurestore.service.MaterializationService;
import com.platform.featurestore.service.OnlineServingService;
import com.platform.featurestore.store.online.RocksDBFeatureStore;
import io.grpc.stub.StreamObserver;
import net.devh.boot.grpc.server.service.GrpcService;
import org.rocksdb.RocksDBException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;

@GrpcService
public class FeatureStoreGrpcService extends FeatureStoreServiceGrpc.FeatureStoreServiceImplBase {

    private static final Logger log = LoggerFactory.getLogger(FeatureStoreGrpcService.class);

    private final OnlineServingService servingService;
    private final MaterializationService materializationService;
    private final RocksDBFeatureStore rocksStore;

    public FeatureStoreGrpcService(OnlineServingService servingService,
                                    MaterializationService materializationService,
                                    RocksDBFeatureStore rocksStore) {
        this.servingService = servingService;
        this.materializationService = materializationService;
        this.rocksStore = rocksStore;
    }

    @Override
    public void getOnlineFeatures(GetFeaturesRequest request,
                                   StreamObserver<GetFeaturesResponse> responseObserver) {
        try {
            int viewVersion = request.getViewVersion() > 0 ? request.getViewVersion() : 1;

            OnlineServingService.BatchServingResult result = servingService.getFeatureVectorsBatch(
                    request.getViewName(), viewVersion, request.getEntityIdsList());

            GetFeaturesResponse response = GetFeaturesResponse.newBuilder()
                    .addAllVectors(result.vectors())
                    .setLatencyUs(result.latencyUs())
                    .addAllWarnings(result.warnings())
                    .build();

            responseObserver.onNext(response);
            responseObserver.onCompleted();
        } catch (Exception e) {
            log.error("GetOnlineFeatures failed", e);
            responseObserver.onError(
                    io.grpc.Status.INTERNAL.withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void putFeatureVector(PutFeatureVectorRequest request,
                                  StreamObserver<PutFeatureVectorResponse> responseObserver) {
        try {
            materializationService.materializeVector(request.getVector());

            PutFeatureVectorResponse response = PutFeatureVectorResponse.newBuilder()
                    .setSuccess(true)
                    .setMessage("Vector materialized successfully")
                    .build();

            responseObserver.onNext(response);
            responseObserver.onCompleted();
        } catch (RocksDBException e) {
            log.error("PutFeatureVector failed", e);
            responseObserver.onNext(PutFeatureVectorResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("Write failed: " + e.getMessage())
                    .build());
            responseObserver.onCompleted();
        }
    }

    @Override
    public void putFeatureVectorBatch(PutFeatureVectorBatchRequest request,
                                       StreamObserver<PutFeatureVectorBatchResponse> responseObserver) {
        try {
            int count = materializationService.materializeVectorBatch(request.getVectorsList());

            PutFeatureVectorBatchResponse response = PutFeatureVectorBatchResponse.newBuilder()
                    .setSuccess(true)
                    .setVectorsWritten(count)
                    .setMessage("Batch materialized successfully")
                    .build();

            responseObserver.onNext(response);
            responseObserver.onCompleted();
        } catch (RocksDBException e) {
            log.error("PutFeatureVectorBatch failed", e);
            responseObserver.onNext(PutFeatureVectorBatchResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("Batch write failed: " + e.getMessage())
                    .build());
            responseObserver.onCompleted();
        }
    }

    @Override
    public void putScalarFeatures(PutScalarFeaturesRequest request,
                                   StreamObserver<PutScalarFeaturesResponse> responseObserver) {
        try {
            for (Feature feature : request.getFeaturesList()) {
                rocksStore.putScalarFeature(request.getEntityType(), request.getEntityId(), feature);
            }

            PutScalarFeaturesResponse response = PutScalarFeaturesResponse.newBuilder()
                    .setSuccess(true)
                    .setFeaturesWritten(request.getFeaturesCount())
                    .setMessage("Scalar features written successfully")
                    .build();

            responseObserver.onNext(response);
            responseObserver.onCompleted();
        } catch (RocksDBException e) {
            log.error("PutScalarFeatures failed", e);
            responseObserver.onNext(PutScalarFeaturesResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("Scalar write failed: " + e.getMessage())
                    .build());
            responseObserver.onCompleted();
        }
    }

    @Override
    public void getViewSchema(GetViewSchemaRequest request,
                               StreamObserver<GetViewSchemaResponse> responseObserver) {
        try {
            int version = request.getVersion() > 0 ? request.getVersion() : 1;
            Optional<ViewSchema> schema = rocksStore.getSchema(request.getViewName(), version);

            if (schema.isPresent()) {
                responseObserver.onNext(GetViewSchemaResponse.newBuilder()
                        .setSchema(schema.get())
                        .build());
            } else {
                responseObserver.onError(
                        io.grpc.Status.NOT_FOUND
                                .withDescription("Schema not found: " + request.getViewName() + ":" + version)
                                .asRuntimeException());
                return;
            }
            responseObserver.onCompleted();
        } catch (RocksDBException e) {
            log.error("GetViewSchema failed", e);
            responseObserver.onError(
                    io.grpc.Status.INTERNAL.withDescription(e.getMessage()).asRuntimeException());
        }
    }
}
