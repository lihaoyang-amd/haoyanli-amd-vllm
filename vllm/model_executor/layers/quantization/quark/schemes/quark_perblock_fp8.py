# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Callable, Optional, cast

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, normalize_e4m3fn_to_e4m3fnuz, requantize_with_max_scale)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.platforms import current_platform
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE, FusedMoEActivationFormat, FusedMoEConfig, FusedMoEMethodBase,
    FusedMoEPermuteExpertsUnpermute, FusedMoEPrepareAndFinalize,
    FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin,
    prepare_moe_fp8_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, all_close_1d, cutlass_block_fp8_supported,
    cutlass_fp8_supported, maybe_create_device_identity,
    normalize_e4m3fn_to_e4m3fnuz, per_tensor_dequantize,
    requantize_with_max_scale)
from vllm.model_executor.parameter import (BlockQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils import has_deep_gemm

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper
__all__ = ["QuarkW8A8Fp8"]


class QuarkBlockFp8(QuarkScheme):

    def __init__(self, weight_config: dict[str, Any],
                 input_config: Optional[dict[str, Any]]):
        self.weight_config = weight_config
        self.input_config = input_config

        self.cutlass_block_fp8_supported = cutlass_block_fp8_supported()
        self.out_dtype = torch.get_default_dtype()

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.use_marlin = (not current_platform.has_device_capability(89)
                            or envs.VLLM_TEST_FORCE_FP8_MARLIN)
        # Disable marlin for rocm
        if current_platform.is_rocm():
            self.use_marlin = False

        # AITER is only supported on ROCm and only for FP8_FNUZ
        # and at the moment are MI300 series
        self.use_aiter_and_is_supported = (current_platform.is_rocm()
                                            and envs.VLLM_ROCM_USE_AITER
                                            and envs.VLLM_ROCM_USE_AITER_LINEAR
                                            and current_platform.is_fp8_fnuz())

        self.weight_block_size = weight_config.get("weight_block_size")
        self.dynamic_input = self.input_config.get("is_dynamic")
        self.block_quant = self.weight_block_size is not None
        self.fp8_linear = Fp8LinearOp(
            # Default to using per_token quantization if cutlass is supported
            use_per_token_if_dynamic=True)

        self.out_dtype = torch.get_default_dtype()
        

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89


    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        maybe_create_device_identity()

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None


        if self.block_quant:
            tp_size = get_tensor_model_parallel_world_size()
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            block_n, block_k = (
                self.weight_block_size[0],
                self.weight_block_size[1],
            )
            # Required by row parallel
            if (tp_size > 1
                    and input_size // input_size_per_partition == tp_size
                    and input_size_per_partition % block_k != 0):
                raise ValueError(
                    f"Weight input_size_per_partition = "
                    f"{input_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}.")
            # Required by column parallel or enabling merged weights
            if (tp_size > 1 and output_size // output_size_per_partition
                    == tp_size) or len(output_partition_sizes) > 1:
                for output_partition_size in output_partition_sizes:
                    if output_partition_size % block_n != 0:
                        raise ValueError(
                            f"Weight output_partition_size = "
                            f"{output_partition_size} is not divisible by "
                            f"weight quantization block_n = {block_n}.")
        # WEIGHT
        weight_dtype = torch.float8_e4m3fn
                        
        weight = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=weight_dtype),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)
        # if self.quant_config.is_checkpoint_fp8_serialized:
        if 1:
            # WEIGHT SCALE
            if self.block_quant:
                assert self.dynamic_input == True
                scale = BlockQuantScaleParameter(
                    data=torch.empty(
                        (output_size_per_partition + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        dtype=torch.float32,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                set_weight_attrs(scale, {"scale_type": "weight_scale"})
                # The weight_scale_inv name is intentional for deepseekv3
                layer.register_parameter("weight_scale_inv", scale)

            # INPUT ACTIVATION SCALE
            if self.dynamic_input:
                layer.register_parameter("input_scale", None)
    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # Pad the weight tensor. This is an optimization on ROCm platform, which
        # can benefit from tensors located far enough from one another in memory
        if (envs.VLLM_ROCM_FP8_PADDING and current_platform.is_rocm()
                and weight.stride(-1) == 1
                and (weight.stride(-2) * weight.element_size()) % 512 == 0):
            num_pad = 256 // weight.element_size()
            weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
            torch.cuda.empty_cache()
        return weight
    def process_weights_after_loading(self, layer: Module) -> None:
        size_k_first = True
        # TODO(rob): refactor block quant into separate class.
        if self.block_quant:
            assert self.dynamic_input == True
            size_k_first = False
            if current_platform.is_fp8_fnuz():
                weight, weight_scale_inv, _ = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        weight=layer.weight,
                        weight_scale=layer.weight_scale_inv)
            else:
                weight = layer.weight.data
                weight_scale_inv = layer.weight_scale_inv.data

            weight = self._maybe_pad_weight(weight)

            # Torch.compile cannot use Parameter subclasses.
            layer.weight = Parameter(weight, requires_grad=False)
            layer.weight_scale_inv = Parameter(weight_scale_inv,
                                               requires_grad=False)
        if self.use_marlin:
            prepare_fp8_layer_for_marlin(layer, size_k_first)
            # Activations not quantized for marlin.
            del layer.input_scale

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_marlin:
            return apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias)

        if self.block_quant:
            assert self.weight_block_size is not None

            return torch.ops.vllm.apply_w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=layer.input_scale,
                bias=bias,
                cutlass_block_fp8_supported=self.cutlass_block_fp8_supported,
                use_aiter_and_is_supported=self.use_aiter_and_is_supported,
            )

        return self.fp8_linear.apply(input=x,
                                     weight=layer.weight,
                                     weight_scale=layer.weight_scale,
                                     out_dtype=self.out_dtype,
                                     input_scale=layer.input_scale,
                                     bias=bias)
