from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict

class MuraLayer(BaseTunerLayer):
    adapter_layer_names = ("lora_A_list", "lora_B_list", "lora_tanh_alpha")
    other_param_names = ("r", "lora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_alpha = {}
        self.scaling = {}

        # For storing vector scale
        self.num_scales = {}
        self.lora_A_list = nn.ParameterDict({})
        self.lora_B_list = nn.ParameterDict({})
        self.lora_tanh_alpha = nn.ParameterDict({})

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self._caches = {}


        self.use_tanh: dict[str, bool] = {}
        self.learnable_tanh: dict[str, bool] = {}
        self.mura_simple: dict[str, bool] = {}

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
            
        self.in_features = in_features 
        self.out_features = out_features
        self.cast_input_dtype_enabled: bool = True
    
    def update_layer(
        self,
        adapter_name,
        r, 
        lora_alpha,
        lora_dropout,
        num_scales: int = 3,
        use_tanh: bool = False,
        learnable_tanh: bool = False,
        mura_simple: bool = True,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if mura_simple:
            self.lora_A_list[adapter_name] = nn.ParameterList([nn.Parameter(torch.randn(self.in_features // (2 ** i), r)) for i in range(num_scales)])
            self.lora_B_list[adapter_name] = nn.ParameterList([nn.Parameter(torch.randn(r, self.out_features // (2 ** i))) for i in range(num_scales)])
        else:
            self.lora_A_list[adapter_name] = nn.ParameterList([nn.Parameter(torch.randn(self.in_features // (2 ** i), r * (2 ** i))) for i in range(num_scales)])
            self.lora_B_list[adapter_name] = nn.ParameterList([nn.Parameter(torch.randn(r * (2 ** i), self.out_features // (2 ** i))) for i in range(num_scales)])
        if learnable_tanh:
            self.lora_tanh_alpha[adapter_name] = nn.Parameter(torch.tensor(0.5))
        else:
            self.lora_tanh_alpha[adapter_name] = 1.0
        
        self.scaling[adapter_name] = lora_alpha / r
        self.reset_parameters(adapter_name)

        self.num_scales[adapter_name] = num_scales
        self.use_tanh[adapter_name] = use_tanh
        self.learnable_tanh[adapter_name] = learnable_tanh
        self.mura_simple[adapter_name] = mura_simple
        
        self.set_adapter(self.active_adapters)
    
    def reset_parameters(self, adapter_name):
        if adapter_name in self.lora_A_list.keys():
            for i in range(len(self.lora_A_list[adapter_name])):
                nn.init.kaiming_uniform_(self.lora_A_list[adapter_name][i], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_list[adapter_name][i])

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            self.scaling[active_adapter] *= scale
    
    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        # DoRA is not supported (yet), check that it's not being used. Don't check "__base__", as this is the
        # placeholder for the base model.
        unique_adapters = {name for name in adapter_names if name != "__base__"}
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)
    

    def _cast_input_dtype(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Whether to cast the dtype of the input to the forward method.

        Usually, we want to enable this to align the input dtype with the dtype of the weight, but by setting
        layer.cast_input_dtype=False, this can be disabled if necessary.

        Enabling or disabling can be managed via the peft.helpers.disable_lora_input_dtype_casting context manager.
        """
        if (not self.cast_input_dtype_enabled) or (x.dtype == dtype):
            return x
        return x.to(dtype=dtype)
        

class Linear(nn.Module, MuraLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        # mura specific parameters
        num_scales: int = 3,
        use_tanh: bool = False,
        learnable_tanh: bool = False,
        mura_simple: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        MuraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_scales=num_scales,
            use_tanh=use_tanh,
            learnable_tanh=learnable_tanh,
            mura_simple=mura_simple,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
    

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return 
        
        for active_adapter in adapter_names:
            if active_adapter in self.lora_A_list.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weights += delta_weight
                
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                    
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data += delta_weight

                self.merged_adapters.append(active_adapter)
    
    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A_list.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                weight.data -= delta_weight

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_A_list[adapter][0].device
        dtype = self.lora_A_list[adapter][0].dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)
        output_tensor = 0
        for i, (A, B) in enumerate(zip(self.lora_A_list[adapter], self.lora_B_list[adapter])):
            if cast_to_fp32:
                A = A.float()
                B = B.float()
            if self.use_tanh[adapter]:
                cur_mat = torch.sum(torch.tanh(self.lora_tanh_alpha[adapter] * (A[..., None] * B[None, ...])), dim=1)
                cur_mat = cur_mat[None, None, :, :]
                cur_mat = torch.nn.functional.interpolate(cur_mat, scale_factor=2**i, mode="bilinear").squeeze()
            else:
                cur_mat = (A@B)[None, None, :, :]
                cur_mat = torch.nn.functional.interpolate(cur_mat, scale_factor=2**i, mode='bilinear').squeeze()
            output_tensor += cur_mat 
        output_tensor = transpose(output_tensor, not self.fan_in_fan_out) * self.scaling[adapter]


        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

        return output_tensor
    
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tenso:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            
            lora_A_list_keys = self.lora_A_list.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_list_keys:
                    continue
                
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, self.lora_A_list[active_adapter][0].dtype)
                
                new_weight = 0
                for i, (A, B) in enumerate(zip(self.lora_A_list[active_adapter], self.lora_B_list[active_adapter])):
                    if self.use_tanh[active_adapter]:
                        cur_mat = torch.sum(torch.tanh(self.lora_tanh_alpha[active_adapter] * (A[..., None] * B[None, ...])), dim=1)
                        cur_mat = cur_mat[None, None, :, :]
                        cur_mat = torch.nn.functional.interpolate(cur_mat, scale_factor=2**i, mode="bilinear").squeeze()
                    else:
                        cur_mat = (A@B)[None, None, :, :]
                        cur_mat = torch.nn.functional.interpolate(cur_mat, scale_factor=2**i, mode='bilinear').squeeze()
                    new_weight += cur_mat
                new_weight = transpose(new_weight, not self.fan_in_fan_out) * scaling
                result = result + F.linear(dropout(x), new_weight, bias=None)
                
            result = result.to(torch_result_dtype)
            
        return result
    
    def __repr__(self) -> str:
        rep = super().__repr__()
        return "mura." + rep


        
                
                    
                    
                    
                