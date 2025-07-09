# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

from typing import Any, Callable, Optional
def _is_col_major(x: torch.Tensor) -> bool:
    assert x.dim() == 3
    b, m, n = x.shape
    return x.stride(0) == m * n and x.stride(1) == 1 and x.stride(2) == m
import torch

import vllm.model_executor.layers.fused_moe  # noqa
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d, normalize_e4m3fn_to_e4m3fnuz, per_tensor_dequantize)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)

__all__ = ["QuarkMoEMethod", "QuarkW8A8Fp8MoEMethod"]


class QuarkMoEMethod(FusedMoEMethodBase):

    @staticmethod
    def get_moe_method(
            quant_config: "QuarkConfig",  # type: ignore # noqa E501 # noqa F821
            module: torch.nn.Module,
            layer_name: str) -> "QuarkMoEMethod":
        layer_quant_config = quant_config._find_matched_config(
            layer_name, module)

        if (layer_quant_config.get("output_tensors")
                or layer_quant_config.get("bias")):
            raise NotImplementedError("Currently, Quark models with "
                                      "output_tensors and bias "
                                      "quantized are not supported")
        weight_config = layer_quant_config.get("weight")
        input_config = layer_quant_config.get("input_tensors")
        block_size = weight_config.get("weight_block_size")
        if block_size is not None:
            return QuarkW8A8Fp8MoEBlockMethod(weight_config, input_config, block_size)

        if quant_config._is_fp8_w8a8(weight_config, input_config):
            return QuarkW8A8Fp8MoEMethod(weight_config, input_config)
        else:
            raise RuntimeError("Unsupported FusedMoe scheme")

class QuarkW8A8Fp8MoEBlockMethod(QuarkMoEMethod):

    def __init__(self, weight_config: dict[str, Any], input_config: dict[str,
                                                                         Any], block_size: list[int, int]):
        from vllm.model_executor.layers.fused_moe import fused_experts
        
        self.weight_quant = weight_config
        self.input_config = input_config
        self.weight_block_size = block_size
        self.block_quant = self.weight_block_size is not None
        self.dynamic_input = self.input_config.get("is_dynamic")
        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.use_marlin = (not current_platform.has_device_capability(89)
                           or envs.VLLM_TEST_FORCE_FP8_MARLIN)
        # Disable marlin for rocm
        if current_platform.is_rocm():
            self.use_marlin = False

        # Check for DeepGemm support.
        self.allow_deep_gemm = False
        if envs.VLLM_USE_DEEP_GEMM:
            if not has_deep_gemm():
                logger.warning_once("Failed to import DeepGemm kernels.")
            elif not self.block_quant:
                logger.warning_once("Model is not block quantized. Not using "
                                    " DeepGemm kernels")
            elif (current_platform.is_cuda()
                  and current_platform.has_device_capability(90)):
                logger.info_once("Using DeepGemm kernels for Fp8MoEMethod.")
                self.allow_deep_gemm = True
            else:
                logger.warning_once(
                    "DeepGemm not supported on the current platform.")

        # Check for CutlassBlockScaledGroupedGemm support.
        self.allow_cutlass_block_scaled_grouped_gemm = False
        if not self.block_quant:
            logger.warning_once("Model is not block quantized. Not using "
                                "CutlassBlockScaledGroupedGemm kernels")
        elif (current_platform.is_cuda()
              and current_platform.has_device_capability(100)):
            logger.info_once(
                "Using CutlassBlockScaledGroupedGemm kernels for Fp8MoEMethod."
            )
            self.allow_cutlass_block_scaled_grouped_gemm = True
        else:
            logger.warning_once(
                "CutlassBlockScaledGroupedGemm not supported on the current "
                "platform.")

        self.topk_indices_dtype = None
        self.fused_experts = functools.partial(  # type: ignore
            fused_experts,
            use_fp8_w8a8=True,
            block_shape=self.weight_block_size,
            allow_deep_gemm=self.allow_deep_gemm,
            allow_cutlass_block_scaled_grouped_gemm=(
                self.allow_cutlass_block_scaled_grouped_gemm))

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        params_dtype = torch.float8_e4m3fn
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            tp_size = get_tensor_model_parallel_world_size()
            block_n, block_k = (
                self.weight_block_size[0],
                self.weight_block_size[1],
            )
            # NOTE: To ensure proper alignment of the block-wise quantization
            # scales, the output_size of the weights for both the gate and up
            # layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}.")
            if (tp_size > 1
                    and intermediate_size_per_partition % block_k != 0):
                # Required by row parallel
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}.")
        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if self.block_quant:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * ((intermediate_size_per_partition + block_n - 1) //
                         block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
            assert self.dynamic_input is True
        if 1:
            # Add the quantization method used (per tensor/grouped/channel)
            # to ensure the weight scales are loaded in properly
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.BLOCK.
                value} if self.block_quant else
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
            # If loading fp8 checkpoint, pass the weight loaders.
            # If loading an fp16 checkpoint, do not (we will quantize in
            #   process_weights_after_loading()
            # if self.quant_config.is_checkpoint_fp8_serialized:
            if 1:
                set_weight_attrs(w13_weight_scale, extra_weight_attrs)
                set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        # INPUT_SCALES
        if self.dynamic_input:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Lazy import to avoid importing triton too early.
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            is_rocm_aiter_moe_enabled, shuffle_weights)

        self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()

        if self.block_quant:
            assert self.dynamic_input is True
            if current_platform.is_fp8_fnuz():
                w13_weight, w13_weight_scale_inv, w13_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight, layer.w13_weight_scale_inv,
                        layer.w13_input_scale)
                w2_weight, w2_weight_scale_inv, w2_input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale_inv,
                        layer.w2_input_scale)
            else:
                w13_weight = layer.w13_weight.data
                w13_weight_scale_inv = layer.w13_weight_scale_inv.data
                w2_weight = layer.w2_weight
                w2_weight_scale_inv = layer.w2_weight_scale_inv

            # torch.compile() cannot use Parameter subclasses.
            layer.w13_weight = Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale_inv = Parameter(w13_weight_scale_inv,
                                                   requires_grad=False)
            layer.w2_weight = Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale_inv = Parameter(w2_weight_scale_inv,
                                                  requires_grad=False)
            if self.rocm_aiter_moe_enabled:
                # reshaping weights is required for aiter moe kernel.
                shuffled_w13, shuffled_w2 = shuffle_weights(
                    layer.w13_weight.data, layer.w2_weight.data)

                layer.w13_weight = torch.nn.Parameter(shuffled_w13,
                                                      requires_grad=False)
                layer.w2_weight = torch.nn.Parameter(shuffled_w2,
                                                     requires_grad=False)

            # DeepGemm scales need to be transposed and aligned.  We try to do
            # it ahead of time for performance reasons.
            if self.allow_deep_gemm:
                # Lazy import to avoid CUDA initialization problems.
                import deep_gemm as dg
                if _is_col_major(layer.w13_weight_scale_inv):
                    layer.w13_weight_scale_inv = \
                        dg.get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv).contiguous()
                if _is_col_major(layer.w2_weight_scale_inv):
                    layer.w2_weight_scale_inv = \
                        dg.get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv).contiguous()
        if self.use_marlin:
            prepare_moe_fp8_layer_for_marlin(layer, False)
            # Activations not quantized for marlin.
            del layer.w13_input_scale
            del layer.w2_input_scale
        if 0: # just for non-block fp8
            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.local_num_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start:start + shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id])
                    layer.w13_weight[expert_id][
                        start:start + shard_size, :], _ = ops.scaled_fp8_quant(
                            dq_weight, max_w13_scales[expert_id])
                    start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales,
                                                        requires_grad=False)
    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        moe: FusedMoEConfig,
    ) -> FusedMoEPermuteExpertsUnpermute:
        from vllm.model_executor.layers.fused_moe import (
            BatchedTritonOrDeepGemmExperts, TritonOrDeepGemmExperts)

        assert not self.use_marlin and not self.rocm_aiter_moe_enabled, (
            "Marlin and ROCm AITER are not supported with all2all yet.")

        if (prepare_finalize.activation_format ==
                FusedMoEActivationFormat.BatchedExperts):
            max_num_tokens_per_rank = (
                prepare_finalize.max_num_tokens_per_rank())
            assert max_num_tokens_per_rank is not None
            logger.debug(
                "BatchedTritonOrDeepGemmExperts(%s): "
                "max_tokens_per_rank=%s, block_size=%s, per_act_token=%s",
                self.__class__.__name__, max_num_tokens_per_rank,
                self.quant_config.weight_block_size, False)
            return BatchedTritonOrDeepGemmExperts(
                max_num_tokens=max_num_tokens_per_rank,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                use_fp8_w8a8=True,
                block_shape=self.quant_config.weight_block_size,
                per_act_token_quant=False,
                allow_deep_gemm=self.allow_deep_gemm,
            )
        else:
            logger.debug(
                "TritonOrDeepGemmExperts(%s): block_size=%s, per_act_token=%s",
                self.__class__.__name__, self.quant_config.weight_block_size,
                False)
            return TritonOrDeepGemmExperts(
                use_fp8_w8a8=True,
                block_shape=self.quant_config.weight_block_size,
                allow_deep_gemm=self.allow_deep_gemm,
            )
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if enable_eplb:
            assert expert_load_view is not None
            assert logical_to_physical_map is not None
            assert logical_replica_count is not None
            assert isinstance(layer, FusedMoE)


        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype,
            enable_eplb=enable_eplb,
            expert_map=expert_map,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
        )

        if self.rocm_aiter_moe_enabled:
            from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (  # noqa: E501
                rocm_aiter_fused_experts)
            return rocm_aiter_fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                use_fp8_w8a8=True,
                apply_router_weight_on_input=apply_router_weight_on_input,
                w1_scale=(layer.w13_weight_scale_inv
                          if self.block_quant else layer.w13_weight_scale),
                w2_scale=(layer.w2_weight_scale_inv
                          if self.block_quant else layer.w2_weight_scale),
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                block_shape=self.quant_config.weight_block_size,
                expert_map=expert_map)
        elif self.use_marlin:
            assert activation == "silu", (
                f"{activation} not supported for Marlin MoE.")
            return torch.ops.vllm.fused_marlin_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                router_logits,
                topk_weights,
                topk_ids,
                quant_type_id=scalar_types.float8_e4m3fn.id,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map)
        else:
            return self.fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                global_num_experts=global_num_experts,
                apply_router_weight_on_input=apply_router_weight_on_input,
                expert_map=expert_map,
                w1_scale=(layer.w13_weight_scale_inv
                          if self.block_quant else layer.w13_weight_scale),
                w2_scale=(layer.w2_weight_scale_inv
                          if self.block_quant else layer.w2_weight_scale),
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
            )

class QuarkW8A8Fp8MoEMethod(QuarkMoEMethod):

    def __init__(self, weight_config: dict[str, Any], input_config: dict[str,
                                                                         Any]):
        self.weight_quant = weight_config
        self.input_quant = input_config

        weight_qscheme = self.weight_quant.get("qscheme")
        input_qscheme = self.input_quant.get("qscheme")
        if not (weight_qscheme == "per_tensor"
                and input_qscheme == "per_tensor"):
            raise ValueError(
                "For FP8 Fused MoE layers, only per-tensor scales "
                "for weights and activations are supported. Found "
                f"{weight_qscheme}, {input_qscheme}")  # noqa E501

        self.static_input_scales = not self.input_quant.get("is_dynamic")

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        # Allocate 2 scales for w1 and w3 respectively.
        # They will be combined to a single scale after weight loading.
        w13_weight_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                         2,
                                                         dtype=torch.float32),
                                              requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        w2_weight_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                        dtype=torch.float32),
                                             requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.static_input_scales:
            w13_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                requires_grad=False)
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Fp8 moe kernels require a single activation scale.
        # We take the max of all the scales in case they differ.
        if self.static_input_scales:
            if (layer.w13_input_scale is None or layer.w2_input_scale is None):
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None.")
            if (not all_close_1d(layer.w13_input_scale)
                    or not all_close_1d(layer.w2_input_scale)):
                logger.warning_once(
                    "Found input_scales that are not equal for "
                    "fp8 MoE layer. Using the maximum across experts "
                    "for each layer. ")
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False)
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False)

        if current_platform.is_fp8_fnuz():
            # Normalize the weights and scales
            w13_weight, w13_weight_scale, w13_input_scale = \
                normalize_e4m3fn_to_e4m3fnuz(
                    layer.w13_weight, layer.w13_weight_scale,
                    layer.w13_input_scale)
            w2_weight, w2_weight_scale, w2_input_scale = \
                normalize_e4m3fn_to_e4m3fnuz(
                    layer.w2_weight, layer.w2_weight_scale,
                    layer.w2_input_scale)
            # Reset the parameter
            layer.w13_weight = torch.nn.Parameter(w13_weight,
                                                  requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(w13_weight_scale,
                                                        requires_grad=False)
            if w13_input_scale is not None:
                layer.w13_input_scale = torch.nn.Parameter(w13_input_scale,
                                                           requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight,
                                                 requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale,
                                                       requires_grad=False)
            if w2_input_scale is not None:
                layer.w2_input_scale = torch.nn.Parameter(w2_input_scale,
                                                          requires_grad=False)

        # Fp8 moe kernel needs single weight scale for w13 per expert.
        # We take the max then dequant and requant each expert.
        assert layer.w13_weight_scale is not None
        shard_size = layer.intermediate_size_per_partition
        max_w13_scales = layer.w13_weight_scale.max(dim=1).values
        for expert_id in range(layer.local_num_experts):
            start = 0
            for shard_id in range(2):
                dq_weight = per_tensor_dequantize(
                    layer.w13_weight[expert_id][start:start + shard_size, :],
                    layer.w13_weight_scale[expert_id][shard_id])
                layer.w13_weight[expert_id][
                    start:start + shard_size, :], _ = ops.scaled_fp8_quant(
                        dq_weight, max_w13_scales[expert_id])
                start += shard_size

        layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales,
                                                    requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `QuarkW8A8Fp8MoEMethod` yet.")

        from vllm.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias)

        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            use_fp8_w8a8=True,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale)
