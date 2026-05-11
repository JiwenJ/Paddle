# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from typing import Literal

import paddle
from paddle.base import framework
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    ShardedStateDict,
    ShardedWeight,
    create_sharded_weight_with_new_local,
)

from .muon import Muon

NormMode = Literal["auto", "row", "col"]

__all__ = ["NorMuon"]


class NorMuon(Muon):
    r"""NorMuon optimizer.

    NorMuon inherits Muon's AdamW fallback and most optimizer plumbing. For
    Muon-selected parameters it inserts neuron-wise second-moment normalization
    after Newton-Schulz orthogonalisation.
    """

    def __init__(
        self,
        learning_rate=0.02,
        parameters=None,
        momentum=0.95,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.01,
        ns_steps=5,
        ns_coeff_type="simple",
        ns_coeffs=None,
        nesterov=True,
        adam_epsilon=1e-9,
        grad_clip=None,
        lr_ratio=None,
        apply_decay_param_fun=None,
        muon_version=1,
        muon_exclude_patterns=None,
        muon_extra_scale_factor=0.2,
        muon_param_info_map=None,
        ns_matmul_dtype=None,
        multi_precision=False,
        name=None,
        normuon_beta2=0.95,
        normuon_epsilon=1e-9,
        normuon_norm_mode: NormMode = "auto",
        normuon_restore_norm=True,
        **kwargs,
    ):
        if not 0.0 <= normuon_beta2 < 1.0:
            raise ValueError(f"Invalid normuon_beta2: {normuon_beta2}")
        if normuon_epsilon <= 0.0:
            raise ValueError(f"Invalid normuon_epsilon: {normuon_epsilon}")
        if normuon_norm_mode not in ("auto", "row", "col"):
            raise ValueError(
                "normuon_norm_mode must be one of 'auto', 'row', or 'col', "
                f"got {normuon_norm_mode!r}."
            )

        super().__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            momentum=momentum,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            ns_coeff_type=ns_coeff_type,
            ns_coeffs=ns_coeffs,
            nesterov=nesterov,
            adam_epsilon=adam_epsilon,
            grad_clip=grad_clip,
            lr_ratio=lr_ratio,
            apply_decay_param_fun=apply_decay_param_fun,
            muon_version=muon_version,
            muon_exclude_patterns=muon_exclude_patterns,
            muon_extra_scale_factor=muon_extra_scale_factor,
            muon_param_info_map=muon_param_info_map,
            ns_matmul_dtype=ns_matmul_dtype,
            multi_precision=multi_precision,
            name=name,
            **kwargs,
        )
        self._default_dict.update(
            {
                "normuon_beta2": normuon_beta2,
                "normuon_epsilon": normuon_epsilon,
                "normuon_norm_mode": normuon_norm_mode,
                "normuon_restore_norm": normuon_restore_norm,
            }
        )

    @staticmethod
    def _normuon_reduce_axis(shape, norm_mode: NormMode):
        if len(shape) < 2:
            raise ValueError(
                f"NorMuon requires a matrix-like parameter, got shape {shape}."
            )
        if norm_mode == "row":
            return len(shape) - 1
        if norm_mode == "col":
            return len(shape) - 2
        return len(shape) - 1 if shape[-2] >= shape[-1] else len(shape) - 2

    @staticmethod
    def _normuon_buffer_shape(shape, norm_mode: NormMode):
        shape = list(shape)
        shape[NorMuon._normuon_reduce_axis(shape, norm_mode)] = 1
        return shape

    def _ensure_accumulators(self, param, use_muon, group):
        super()._ensure_accumulators(param, use_muon, group)
        if not use_muon or (
            self._moment2_acc_str in self._accumulators
            and param.name in self._accumulators[self._moment2_acc_str]
        ):
            return

        self._add_accumulator(
            self._moment2_acc_str,
            param,
            dtype=paddle.float32,
            fill_value=0.0,
            shape=self._normuon_buffer_shape(
                getattr(param, "original_shape", param.shape),
                group.get("normuon_norm_mode", "auto"),
            ),
            type=framework.core.VarDesc.VarType.DENSE_TENSOR,
        )

    def _normuon_normalize(self, param, orthogonal_update):
        moment2 = self._get_accumulator(self._moment2_acc_str, param)
        beta2 = self._default_dict.get("normuon_beta2", 0.95)
        epsilon = self._default_dict.get("normuon_epsilon", 1e-9)
        axis = self._normuon_reduce_axis(
            orthogonal_update.shape,
            self._default_dict.get("normuon_norm_mode", "auto"),
        )

        update_f32 = orthogonal_update.astype(paddle.float32)
        mean_square = paddle.mean(
            paddle.square(update_f32), axis=axis, keepdim=True
        )
        if list(mean_square.shape) != list(moment2.shape):
            raise RuntimeError(
                "NorMuon second-moment accumulator shape mismatch: "
                f"expected {list(mean_square.shape)}, got {list(moment2.shape)}."
            )

        paddle.assign(paddle.lerp(moment2, mean_square, 1.0 - beta2), moment2)
        normalized_update = update_f32 / paddle.maximum(
            paddle.sqrt(moment2),
            paddle.full(moment2.shape, epsilon, dtype=moment2.dtype),
        )

        if self._default_dict.get("normuon_restore_norm", True):
            old_norm = paddle.linalg.norm(update_f32)
            new_norm = paddle.maximum(
                paddle.linalg.norm(normalized_update),
                paddle.full([], epsilon, dtype=paddle.float32),
            )
            normalized_update *= old_norm / new_norm

        return normalized_update.astype(orthogonal_update.dtype)

    def _muon_update(
        self,
        param,
        grad,
        lr,
        momentum_buffer,
        momentum_beta,
        ns_steps,
        nesterov,
        epsilon,
        weight_decay,
        version,
    ):
        param_shape = getattr(param, "original_shape", param.shape)
        param_info = self._muon_param_info_map.get(param.name)

        with paddle.no_grad():
            grad_f32 = (
                grad.astype(momentum_buffer.dtype)
                if grad.dtype != momentum_buffer.dtype
                else grad
            )
            paddle.assign(
                paddle.lerp(momentum_buffer, grad_f32, 1.0 - momentum_beta),
                momentum_buffer,
            )
            update_buffer = (
                paddle.lerp(grad_f32, momentum_buffer, momentum_beta)
                if nesterov
                else momentum_buffer
            )

            def ortho_fn(m):
                return Muon._scaling_fn(
                    Muon._zeropower_via_newtonschulz5(
                        m,
                        steps=ns_steps,
                        eps=epsilon,
                        ns_coeff_type=self._ns_coeff_type,
                        ns_matmul_dtype=self._ns_matmul_dtype,
                    ),
                    version,
                    self._muon_extra_scale_factor,
                )

            matrix = update_buffer.reshape(param_shape)
            orthogonal_update = (
                param_info.split_concat_func(matrix, ortho_fn)
                if param_info is not None
                and param_info.split_concat_func is not None
                else ortho_fn(matrix)
            )
            orthogonal_update = self._normuon_normalize(
                param, orthogonal_update
            )

            find_master = param.name in self._master_weights
            master_weight = (
                self._master_weights[param.name] if find_master else None
            )

            with_decay = True
            if (
                self._apply_decay_param_fun is not None
                and not self._apply_decay_param_fun(param.name)
            ):
                with_decay = False
            if with_decay and weight_decay > 0:
                if find_master:
                    master_weight.scale_(1.0 - lr * weight_decay)
                else:
                    param.scale_(1.0 - lr * weight_decay)

            final_step = orthogonal_update * lr
            if find_master:
                master_weight.subtract_(final_step)
                paddle.assign(master_weight.astype(param.dtype), param)
            else:
                param.subtract_(final_step.astype(param.dtype))

    def sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict,
    ) -> ShardedStateDict:
        """Build a sharded optimizer state dict for flex checkpoint."""
        _FP32_MASTER = "fp32_master_0"
        _optimizer_scalar_names = [
            "beta1_pow_acc_0",
            "beta2_pow_acc_0",
        ]
        _optimizer_vector_names = [
            "moment1_0",
            "moment2_0",
        ]

        def _split_state_name(vname):
            if _FP32_MASTER in vname:
                return tuple(vname.split("_" + _FP32_MASTER + "_", 1))
            for suffix in _optimizer_scalar_names + _optimizer_vector_names:
                if vname.endswith(suffix):
                    return vname[: -(len(suffix) + 1)], suffix
            raise ValueError(
                f"Cannot parse optimizer state variable name: {vname!r}"
            )

        model_sharded_state_dict = dict(
            sorted(model_sharded_state_dict.items())
        )

        static_to_struct = {}
        for struct_name, sw in model_sharded_state_dict.items():
            local_name = sw.local_tensor.name
            if local_name not in static_to_struct:
                static_to_struct[local_name] = struct_name

        optimizer_state_dict = self.state_dict()
        master_weights = optimizer_state_dict.pop("master_weights", None)
        optimizer_state_dict.pop("LR_Scheduler", None)

        sharded_state: ShardedStateDict = {}

        for key, tensor in optimizer_state_dict.items():
            static_name, state_type = _split_state_name(key)
            struct_name = static_to_struct[static_name]
            sharded_param = model_sharded_state_dict[struct_name]
            unified_name = f"{struct_name}.{state_type}"

            if state_type in _optimizer_vector_names:
                target_shape = sharded_param.local_shape
                tensor_shape = tuple(tensor.shape)
                if tensor.is_dist():
                    if tensor_shape == tuple(sharded_param.global_shape):
                        sharded_state[unified_name] = ShardedWeight(
                            key=unified_name,
                            local_tensor=tensor,
                            local_shape=tensor.shape,
                            global_shape=tensor.shape,
                            global_offset=sharded_param.global_offset,
                        )
                    elif state_type == "moment2_0":
                        sharded_state[unified_name] = ShardedWeight(
                            key=unified_name,
                            local_tensor=tensor,
                            local_shape=tensor_shape,
                            global_shape=tensor_shape,
                            global_offset=tuple(0 for _ in tensor_shape),
                        )
                    else:
                        raise ValueError(
                            "Unexpected distributed optimizer state shape "
                            f"mismatch: {key} has shape {tensor_shape}, "
                            f"expected {tuple(sharded_param.global_shape)}."
                        )
                elif tensor_shape == tuple(target_shape):
                    sharded_state[unified_name] = (
                        create_sharded_weight_with_new_local(
                            unified_name, tensor, sharded_param
                        )
                    )
                elif tensor.numel() == paddle.to_tensor(
                    list(target_shape)
                ).prod().item():
                    sharded_state[unified_name] = (
                        create_sharded_weight_with_new_local(
                            unified_name,
                            tensor.reshape(target_shape),
                            sharded_param,
                        )
                    )
                elif state_type == "moment2_0":
                    sharded_state[unified_name] = ShardedWeight(
                        key=unified_name,
                        local_tensor=tensor,
                        local_shape=tensor_shape,
                        global_shape=tensor_shape,
                        global_offset=tuple(0 for _ in tensor_shape),
                    )
                else:
                    raise ValueError(
                        "Unexpected optimizer state shape mismatch: "
                        f"{key} has shape {tensor_shape}, expected "
                        f"{tuple(target_shape)}."
                    )
            else:
                sharded_state[unified_name] = ShardedWeight(
                    key=unified_name,
                    local_tensor=tensor,
                    local_shape=(1,),
                    global_shape=(1,),
                    global_offset=(0,),
                )

        if master_weights:
            for weight_key, tensor in master_weights.items():
                struct_name = static_to_struct[weight_key]
                sharded_param = model_sharded_state_dict[struct_name]
                unified_name = f"{struct_name}.w_0"

                if tensor.is_dist():
                    sharded_state[unified_name] = ShardedWeight(
                        key=unified_name,
                        local_tensor=tensor,
                        local_shape=tensor.shape,
                        global_shape=tensor.shape,
                        global_offset=sharded_param.global_offset,
                    )
                else:
                    sharded_state[unified_name] = (
                        create_sharded_weight_with_new_local(
                            unified_name, tensor, sharded_param
                        )
                    )

        return sharded_state

