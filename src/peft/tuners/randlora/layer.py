# Copyright 2023-present the HuggingFace Inc. team.
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

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict

class UniqueBaseGrad(torch.autograd.Function):
    #Memory efficent for a unique base 
    @staticmethod    
    def forward(ctx, randbasis_A, randlora_lambda, randlora_gamma):
        Out = randlora_lambda[:, :, None] * randbasis_A * randlora_gamma[None, ]
        ctx.save_for_backward(randbasis_A, randlora_lambda, randlora_gamma)
        return Out 
    
    @staticmethod
    def backward(ctx, grad_output):
        randbasis_A, randlora_lambda, randlora_gamma = ctx.saved_tensors
        randbasis_A, randlora_lambda, randlora_gamma = randbasis_A.to(grad_output.dtype), randlora_lambda.to(grad_output.dtype), randlora_gamma.to(grad_output.dtype)
        grad_randlora_lambda = torch.einsum('kbj,kvj,bj->kb', grad_output, randbasis_A, randlora_gamma)
        grad_randlora_gamma = torch.einsum('kbj,kvj,kb->bj', grad_output, randbasis_A, randlora_lambda)
        return None, grad_randlora_lambda, grad_randlora_gamma

class RandLoraLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("randlora_lambda", "randlora_gamma")
    other_param_names = ("randbasis_A", "randbasis_B", "scaling")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.scaling = {}
        self.randlora_dropout = nn.ModuleDict({})

        # For storing vector scale
        self.randlora_lambda = nn.ParameterDict({})
        self.randlora_gamma = nn.ParameterDict({})
        self.randlora_m = nn.ParameterDict({})
        self.randlora_cache_norm_dora = nn.ParameterDict({})

        # Stores a reference to the randbasis_A/B BufferDict.
        # Set to `None` otherwise to avoid computation with random weights
        self.randbasis_A: Optional[BufferDict] = None
        self.randbasis_B: Optional[BufferDict] = None

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name,
        randbasis_A: BufferDict,
        randbasis_B: BufferDict,
        r,
        randlora_alpha,
        randlora_dropout,
        init_weights,
        use_dora,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        if randlora_dropout > 0.0:
            randlora_dropout_layer = nn.Dropout(p=randlora_dropout)
        else:
            randlora_dropout_layer = nn.Identity()

        self.randlora_dropout.update(nn.ModuleDict({adapter_name: randlora_dropout_layer}))

        # Actual trainable parameters
        n = min(self.in_features, self.out_features) / r
        self.n = int(n) if n.is_integer() else int(n) + 1 #Full rank
        self.randlora_lambda[adapter_name] = nn.Parameter(torch.zeros(r, self.n), requires_grad=True)
        self.randlora_gamma[adapter_name] = nn.Parameter(torch.ones(self.n, min(self.out_features, self.in_features))/max(self.out_features, self.in_features), requires_grad=True)
        if use_dora:
            self.randlora_m[adapter_name] = nn.Parameter(torch.zeros(self.in_features, 1), requires_grad=True)
            with torch.no_grad():
                self.randlora_m[adapter_name].data = torch.linalg.norm(self.weight.detach(), dim=1).unsqueeze(1).detach()
                self.randlora_cache_norm_dora[adapter_name] = self.randlora_m[adapter_name].data

        # self.scaling[adapter_name] = randlora_alpha / r / math.sqrt(self.n)#* 10#/ math.sqrt(self.n)
        self.scaling[adapter_name] = randlora_alpha / r

        # non trainable references to randbasis_A/B buffers
        self.randbasis_A = randbasis_A
        self.randbasis_B = randbasis_B
        if adapter_name not in randbasis_A:
            # This means that this is not the first RandLora adapter. We have to add an entry in the dict for this adapter.
            if len(self.randbasis_A) < 1:
                raise ValueError(
                    "The `randbasis_A` and `randbasis_B` buffers are empty. This should not happen. Please report this issue."
                )
            # we can take any of the existing adapter's parameters, as they should all be identical
            randbasis_A_param = list(self.randbasis_A.values())[0]
            randbasis_B_param = list(self.randbasis_B.values())[0]

            error_tmpl = (
                "{} has a size of {} but {} or greater is required; this probably happened because an additional RandLora "
                "adapter was added after the first one with incompatible shapes."
            )
            # check input size
            if randbasis_A_param.shape[-1] < self.in_features:
                raise ValueError(error_tmpl.format("randbasis_A", randbasis_A_param.shape[1], self.in_features))
            # check output size
            if randbasis_B_param.shape[0] < self.out_features:
                raise ValueError(error_tmpl.format("randbasis_B", randbasis_B_param.shape[0], self.out_features))
            # check r
            error_tmpl = (
                "{} has a size of {} but {} or greater is required; this probably happened because an additional RandLora "
                "adapter with a lower rank was added after the first one; loading the adapters "
                "in reverse order may solve this."
            )
            if randbasis_A_param.shape[0] < self.r[adapter_name]:
                raise ValueError(error_tmpl.format("randbasis_A", randbasis_A_param.shape[0], self.r[adapter_name]))
            if randbasis_B_param.shape[1] < self.r[adapter_name]:
                raise ValueError(error_tmpl.format("randbasis_B", randbasis_B_param.shape[1], self.r[adapter_name]))

            self.randbasis_A[adapter_name] = randbasis_A_param
            self.randbasis_B[adapter_name] = randbasis_B_param

        if init_weights:
            self.reset_randlora_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_randlora_parameters(self, adapter_name):
        if adapter_name in self.randlora_lambda.keys():
            with torch.no_grad():
                nn.init.zeros_(self.randlora_lambda[adapter_name])
                nn.init.zeros_(self.randlora_gamma[adapter_name]).fill_(1/max(self.randlora_gamma[adapter_name].shape))


class Linear(nn.Linear, RandLoraLayer):
    # RandLora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        randbasis_A: BufferDict,
        randbasis_B: BufferDict,
        adapter_name: str,
        r: int = 0,
        randlora_alpha: int = 0,
        randlora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_weights: bool = True,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        RandLoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, randbasis_A, randbasis_B, r, randlora_alpha, randlora_dropout, init_weights, use_dora)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.randlora_lambda.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    if active_adapter in self.randlora_m.keys():
                        norm = torch.linalg.norm(orig_weight, dim=1, keepdim=True)
                        self.randlora_cache_norm_dora[active_adapter] = norm
                        orig_weight *= (self.randlora_m[active_adapter] / norm).view(1, -1)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.randlora_lambda.keys():
                if active_adapter in self.randlora_m.keys():
                    ori_weight = self.get_base_layer().weight.data
                    delta = self.get_delta_weight(active_adapter)
                    if active_adapter in self.randlora_m.keys():
                        norm = self.randlora_cache_norm_dora[active_adapter]
                    else:
                        norm = self.randlora_m[active_adapter].data
                        
                    self.get_base_layer().weight.data = \
                        ori_weight * (norm / self.randlora_m[active_adapter]).view(1, -1) - delta
                else:
                    self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_scaled_bases(self, adapter) -> tuple[torch.Tensor, torch.Tensor, torch.dtype]:
        """
        Performs scaling on the smallest random base (randbasis_A) and returns randbasis_A and randbasis_B in the correct order
        to fit the target layers' dimensions
        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        
        randbasis_A = self.randbasis_A[adapter]
        randbasis_B = self.randbasis_B[adapter]

        device = randbasis_B.device
        dtype = randbasis_B.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        randlora_lambda = self.randlora_lambda[adapter]
        randlora_gamma = self.randlora_gamma[adapter]

        if cast_to_fp32:
            randbasis_A = randbasis_A.float()
            randbasis_B = randbasis_B.float()
            randlora_lambda = randlora_lambda.float()
            randlora_gamma = randlora_gamma.float()

        #The trainable paramters are always applied to randbasis_A, the smallest basis.
        min_dim, max_dim = min(self.out_features, self.in_features), max(self.out_features, self.in_features)
        
        # As adapted layers may have different shapes and RandLora contains a single shared pair of A and B matrices,
        # we initialize these matrices with the largest required size for each dimension.
        # During the forward pass, required submatrices are sliced out from the shared randbasis_A and randbasis_B.        
        sliced_A = randbasis_A[:, : self.n, : min_dim]
        sliced_B = randbasis_B[: max_dim, : self.n, :]
        
        #Flattening the matrices over the rank and number of bases dimensions is more memory efficient
        update_B = sliced_B.flatten(start_dim=1)
        update_A = UniqueBaseGrad.apply(sliced_A, randlora_lambda, randlora_gamma).flatten(end_dim=1)
        if min_dim == self.in_features:
            return update_A, update_B, dtype            
        return update_B.T, update_A.T, dtype
        
    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """

        update_B, update_A, dtype = self.get_scaled_bases(adapter)
        
        update = (update_B.T @ update_A.T).T
        output_tensor = transpose(update, self.fan_in_fan_out)

        if dtype != self.randbasis_B[adapter].dtype:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            # TODO: why?, taken from the VeRA implementation
            self.randlora_lambda[adapter].data = randlora_lambda.to(dtype)
            self.randlora_gamma[adapter].data = randlora_gamma.to(dtype)

        scaling = self.scaling[adapter]
        return output_tensor * scaling

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.randlora_lambda.keys():
                    continue                
                dropout = self.randlora_dropout[active_adapter]
                update_B, update_A, _ = self.get_scaled_bases(active_adapter)
                x = x.to(update_A.dtype)
                scaling = self.scaling[active_adapter]
                if active_adapter in self.randlora_m.keys():
                    update = update_A @ update_B * scaling
                    weight_update_norm = torch.linalg.norm(update + self.weight, dim=1, keepdim=True).detach()
                    lora_result = F.linear(F.linear(dropout(x), update_B), update_A) * scaling 
                    result = (result + lora_result) * (self.randlora_m[active_adapter] / weight_update_norm).view(1, -1)
                else:
                    result = result + F.linear(F.linear(dropout(x), update_B), update_A) * scaling

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "randlora." + rep