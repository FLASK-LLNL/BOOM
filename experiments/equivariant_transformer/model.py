################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from typing import Optional, Tuple
import torch
import torch.nn as nn
import re
from torchmdnet.models.model import create_model
from torchmdnet.models.utils import act_class_mapping
from torchmdnet.models.output_modules import (
    OutputModel,
    Scalar,
    EquivariantDipoleMoment,
    ElectronicSpatialExtent,
    DipoleMoment,
)
from torchmdnet.models.torchmd_et import TorchMD_ET


class ScalarWithDropout(OutputModel):
    def __init__(
        self,
        hidden_channels,
        activation="silu",
        allow_prior_model=True,
        reduce_op="sum",
        dropout=0.5,
        dtype=torch.float,
    ):
        super(ScalarWithDropout, self).__init__(
            allow_prior_model=allow_prior_model, reduce_op=reduce_op
        )
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2, dtype=dtype),
            act_class(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1, dtype=dtype),
        )

        self.reset_parameters()

        # TODO: Add custom "reduce" function for attention reduction to extend "sum" or "mean"

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[1].weight)
        self.output_network[1].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[-1].weight)
        self.output_network[-1].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        return self.output_network(x)


class ETModel(nn.Module):
    def __init__(
        self,
        model_path,
        device,
        derivative=False,
        reduce_op="sum",
        dropout=0.1,
        max_num_neighbors=96,
        full_model=False,
        self_supervised=False,
        random_init=False,
        training_mean=0.0,
        training_std=1.0,
        is_dipole_moment=False,
        is_spatial_extent=False,
    ):
        super(ETModel, self).__init__()

        training_mean = torch.scalar_tensor(training_mean)
        training_std = torch.scalar_tensor(training_std)

        self.register_buffer("training_mean", training_mean)
        self.register_buffer("training_std", training_std)
        ckpt = torch.load(model_path)
        self.args = ckpt["hyper_parameters"]
        # breakpoint()
        # print(self.args)
        self.args["reduce_op"] = reduce_op
        self.args["derivavtive"] = derivative
        self.args["max_num_neighbors"] = max_num_neighbors
        # self.args["output_model"] = "Scalar"

        self.model = create_model(self.args)
        state_dict = {
            re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()
        }

        if "prior_model.initial_atomref" in state_dict:
            state_dict["prior_model.0.initial_atomref"] = state_dict[
                "prior_model.initial_atomref"
            ]
            del state_dict["prior_model.initial_atomref"]
        if "prior_model.atomref.weight" in state_dict:
            state_dict["prior_model.0.atomref.weight"] = state_dict[
                "prior_model.atomref.weight"
            ]
            del state_dict["prior_model.atomref.weight"]
        # self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.prior_model = None

        if dropout > 0:
            self.model.output_model = ScalarWithDropout(
                self.args["embedding_dimension"],
                reduce_op=reduce_op,
                dropout=dropout,
            )
        else:
            if is_dipole_moment:
                self.model.output_model = EquivariantDipoleMoment(
                    self.args["embedding_dimension"], reduce_op=reduce_op
                )
            elif is_spatial_extent:
                self.model.output_model = ElectronicSpatialExtent(
                    self.args["embedding_dimension"], reduce_op=reduce_op
                )
            else:
                self.model.output_model = Scalar(
                    self.args["embedding_dimension"], reduce_op=reduce_op
                )

        self.model.output_model.reset_parameters()

        if random_init:
            self.model.representation_model.reset_parameters()

        for params in self.model.representation_model.parameters():
            params.requires_grad = full_model or random_init

        self.self_supervised = self_supervised
        if self.self_supervised:
            hidden_channels = self.model.representation_model.hidden_channels
            self.output_vector = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.SiLU(),
                nn.Linear(hidden_channels // 2, 1),
            )

    def forward(self, z, pos, batch=None):
        assert z.ndim == 1
        assert pos.ndim == 2
        batch = torch.zeros_like(z) if batch is None else batch
        x, v, z, pos, batch = self.model.representation_model(z, pos, batch)
        x = self.model.output_model.pre_reduce(x, v, z, pos, batch)

        x = x * self.training_std
        y = self.model.output_model.reduce(x, batch)

        y = y + self.training_mean

        y = self.model.output_model.post_reduce(y)
        if self.self_supervised:
            v = self.output_vector(v)
            v = pos.unsqueeze(2) + v
        return y, v

    @torch.no_grad()
    def prediction(self, z, pos, batch):
        assert z.ndim == 1
        assert pos.ndim == 2
        batch = torch.zeros_like(z) if batch is None else batch
        return self.model(z, pos, batch)


class Loss_EMA:
    def __init__(self, beta):
        self.beta = beta
        self.ema = None
        self.counter = 0

    def update(self, x):
        if self.ema is None:
            self.ema = x
        else:
            self.ema = self.beta * self.ema + (1 - self.beta) * x
        self.counter += 1
        return self.ema

    def get_counter(self):
        return self.counter

    def stop_criteria(self, loss):
        return self.counter > 1000 and abs(self.ema - loss) < 1e-6


if __name__ == "__main__":
    # print("Hello")
    device = "cuda:0"
    model = ETModel("pre-trained.ckpt", "cuda:0")
    print(model.args)
    z = torch.tensor([1, 1, 8], dtype=torch.long).to(device)
    pos = torch.rand(3, 3).to(device)
    energy, forces = model(z, pos)
    print(energy)
    print(forces)
