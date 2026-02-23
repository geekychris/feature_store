"""
CUDA/C code generator for XGBoost tree ensemble models.

Converts trained XGBoost models into standalone CUDA or C source code
that scores batches of feature vectors in parallel.

Usage:
    from cuda_codegen import CudaCodeGenerator

    gen = CudaCodeGenerator("models/model.ubj")
    gen.generate_cuda("generated/scoring_kernel.cu")
    gen.generate_cpu("generated/scoring_kernel.c")
"""

from .generator import CudaCodeGenerator, ModelInfo

__all__ = ["CudaCodeGenerator", "ModelInfo"]
