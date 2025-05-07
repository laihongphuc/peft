import warnings
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class KronFTLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("fourierft_spectrum",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("fourierft_n_frequency", "fourierft_scaling", "fourierft_random_loc_seed")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.fourierft_n_frequency = {}
        self.fourierft_n_pack = {}
        self.fourierft_scaling = {}
        self.fourierft_spectrum = nn.ParameterDict({})
        self.indices = {}
        self.gate_indices = {}
        self.fourierft_random_loc_seed = {}
        self.fourierft_B_list = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            self.in_features, self.out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

    def update_layer(self, adapter_name, n_frequency, n_pack, scaling, init_weights, init_eye, random_loc_seed):
        assert n_frequency % (n_pack) == 0, "n_frequency must be divisible by n_pack"
        if n_frequency <= 0:
            raise ValueError(f"`n_frequency` should be a positive integer value but the value passed is {n_frequency}")
        if n_frequency > self.in_features * self.out_features:
            raise ValueError(
                f"`n_frequency` should be less than or equal to the product of the input and output dimensions "
                f"but the value passed is {n_frequency} and the product is {self.in_features * self.out_features}"
            )
        self.fourierft_n_frequency[adapter_name] = n_frequency
        self.fourierft_n_pack[adapter_name] = n_pack
        self.fourierft_random_loc_seed[adapter_name] = random_loc_seed
        # random indices for each pack matrix
        indices_list = [torch.randperm(
            self.out_features * self.in_features // (n_pack * n_pack),
            generator=torch.Generator().manual_seed(self.fourierft_random_loc_seed[adapter_name] + i),
        )[:n_frequency // (n_pack )] for i in range(n_pack)]
        in_features = self.in_features // n_pack
        # indices list: [n_pack, n_frequency // (n_pack)]
        self.indices[adapter_name] = [torch.stack(
            [indices_list[i] // in_features, indices_list[i] % in_features], dim=0
        ) for i in range(n_pack)]
        self.fourierft_scaling[adapter_name] = scaling
        # Actual trainable parameters
        self.fourierft_spectrum[adapter_name] = nn.ParameterList(
            [nn.Parameter(torch.randn(n_frequency // (n_pack)), requires_grad=True) for _ in range(n_pack)]
        )
        if init_eye:
            self.fourierft_B_list[adapter_name] = nn.ParameterList(
                [nn.Parameter(torch.eye(n_pack, n_pack), requires_grad=True) for _ in range(n_pack)]
            )
        else:
            self.fourierft_B_list[adapter_name] = nn.ParameterList(
                [nn.Parameter(torch.zeros(n_pack, n_pack), requires_grad=True) for _ in range(n_pack)]
            )
        # Gating mechanism
        n_gate_parameter = n_pack * self.in_features // 10
        gate_indices = torch.randperm(
            n_pack * self.in_features,
            generator=torch.Generator().manual_seed(self.fourierft_random_loc_seed[adapter_name]),
        )[:n_gate_parameter] 
        self.gate_indices[adapter_name] = torch.stack(
            [gate_indices // self.in_features, gate_indices % self.in_features], dim=0
        )

        if init_weights:
            self.reset_fourier_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    @torch.no_grad()
    def reset_fourier_parameters(self, adapter_name):
        if adapter_name in self.fourierft_spectrum.keys():
            nn.init.zeros_(self.fourierft_spectrum[adapter_name])

    def get_delta_weight(self, adapter) -> torch.Tensor:
        device = self.fourierft_spectrum[adapter][0].device
        dtype = self.fourierft_spectrum[adapter][0].dtype
        spectrum = self.fourierft_spectrum[adapter]
        indices = [ind.to(device) for ind in self.indices[adapter]]
        n_pack = self.fourierft_n_pack[adapter]
        dense_spectrum = [torch.zeros(self.out_features // n_pack, self.in_features // n_pack, device=device, dtype=dtype) for _ in range(n_pack)]
        delta_weight = 0
        for i in range(n_pack):
            dense_spectrum[i][indices[i][0, :], indices[i][1, :]] = spectrum[i]
            # delta_weight += torch.kron(self.fourierft_B_list[adapter][i], torch.fft.ifft2(dense_spectrum[i]).real)
            delta_weight += torch.kron(torch.fft.ifft2(dense_spectrum[i]).real, self.fourierft_B_list[adapter][i])
        delta_weight *= self.fourierft_scaling[adapter]
        return delta_weight
    

class KronFTLinear(nn.Module, KronFTLayer):
    # KronFT implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        n_frequency: int = 1000,
        n_pack: int = 2,
        scaling: float = 150.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_weights: Union[bool, str] = False,
        init_eye: Union[bool, str] = False,
        random_loc_seed: int = 777,
        **kwargs,
    ) -> None:
        super().__init__()
        KronFTLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, n_frequency, n_pack, scaling, init_weights, init_eye, random_loc_seed)

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
            if active_adapter in self.fourierft_spectrum.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.fourierft_spectrum.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        return super().get_delta_weight(adapter)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
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
                if active_adapter not in self.fourierft_spectrum.keys():
                    continue
                
                adapter = activate_adapter

                scaling = self.fourierft_scaling[adapter]
                spectrum = self.fourierft_spectrum[adapter]
                gate_parameter = self.fourierft_gate[adapter]
                indices = [ind.to(device) for ind in self.indices[adapter]]
                gate_indices = self.gate_indices[adapter]
                n_pack = self.fourierft_n_pack[adapter]

                dense_spectrum = [torch.zeros(self.out_features // n_pack, self.in_features // n_pack, device=device, dtype=previous_dtype) for _ in range(n_pack)]


                fourierft_gate = torch.zeros(n_pack, self.in_features, device=device, dtype=previous_dtype)
                fourierft_gate[gate_indices[0, :], gate_indices[1, :]] = gate_parameter
                fourierft_gate = torch.fft.ifft2(fourierft_gate).real

                gate_scores = torch.einsum('bld,ed->ble', x, fourierft_gate)
                gate_weights = F.softmax(gate_scores, dim=-1)  # (batch_size, in_features, n_pack)

                output_list = [0] * n_pack
                for i in range(n_pack):
                    dense_spectrum[i][indices[i][0, :], indices[i][1, :]] = spectrum[i]
                    delta_weight = scaling * torch.kron(torch.fft.ifft2(dense_spectrum[i]).real, self.fourierft_B_list[adapter][i])

                    output = F.linear(x, delta_weight)
                    output_list[i] = output


                stacked_output = torch.stack(output_list, dim= 0)  # (n_pack, batch_size, seq_len, hidden_dim)
                stacked_output = stacked_output.permute(1, 2, 0, 3) # (batch_size, seq_len, n_pack, hidden_dim)

                gate_weights = gate_weights.unsqueeze(-1)

                final_output = (stacked_output * gate_weights).sum(dim=2)

                # final_output = torch.mean(final_output, dim= 0)
                result = result + final_output

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "kronft." + rep