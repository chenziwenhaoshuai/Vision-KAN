import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *


class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)  # Using Xavier Uniform initialization


class ReflectionalSwitchFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            exponent: int = 2,
            denominator: float = 0.33,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator  # or (grid_max - grid_min) / (num_grids - 1)
        # self.exponent = exponent
        self.inv_denominator = 1 / self.denominator  # Cache the inverse of the denominator

    def forward(self, x):
        diff = (x[..., None] - self.grid)
        diff_mul = diff.mul(self.inv_denominator)
        diff_tanh = torch.tanh(diff_mul)
        diff_pow = -diff_tanh.mul(diff_tanh)
        diff_pow += 1
        # diff_pow *= 0.667
        return diff_pow  # Replace pow with multiplication for squaring


class FasterKANLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            exponent: int = 2,
            denominator: float = 0.33,
            use_base_update: bool = True,
            base_activation=F.silu,
            spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = ReflectionalSwitchFunction(grid_min, grid_max, num_grids, exponent, denominator)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        # self.use_base_update = use_base_update
        # if use_base_update:
        #    self.base_activation = base_activation
        #    self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, time_benchmark=False):
        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x)).view(x.shape[0], -1)
            # print("spline_basis:", spline_basis.shape)
        else:
            spline_basis = self.rbf(x).view(x.shape[0], -1)
            # print("spline_basis:", spline_basis.shape)
        # print("-------------------------")
        # ret = 0
        ret = self.spline_linear(spline_basis)
        # print("spline_basis.shape[:-2]:", spline_basis.shape[:-2])
        # print("*spline_basis.shape[:-2]:", *spline_basis.shape[:-2])
        # print("spline_basis.view(*spline_basis.shape[:-2], -1):", spline_basis.view(*spline_basis.shape[:-2], -1).shape)
        # print("ret:", ret.shape)
        # print("-------------------------")
        # if self.use_base_update:
        # base = self.base_linear(self.base_activation(x))
        # print("self.base_activation(x):", self.base_activation(x).shape)
        # print("base:", base.shape)
        # print("@@@@@@@@@")
        # ret += base
        return ret

        # spline_basis = spline_basis.reshape(x.shape[0], -1)  # Reshape to [batch_size, input_dim * num_grids]
        # print("spline_basis:", spline_basis.shape)

        # spline_weight = self.spline_weight.view(-1, self.spline_weight.shape[0])  # Reshape to [input_dim * num_grids, output_dim]
        # print("spline_weight:", spline_weight.shape)

        # spline = torch.matmul(spline_basis, spline_weight)  # Resulting shape: [batch_size, output_dim]

        # print("-------------------------")
        # print("Base shape:", base.shape)
        # print("Spline shape:", spline.shape)
        # print("@@@@@@@@@")


class FasterKAN(nn.Module):
    def __init__(
            self,
            layers_hidden: List[int],
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            exponent: int = 2,
            denominator: float = 0.33,
            use_base_update: bool = True,
            base_activation=F.silu,
            spline_weight_init_scale: float = 0.667,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FasterKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                exponent=exponent,
                denominator=denominator,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x