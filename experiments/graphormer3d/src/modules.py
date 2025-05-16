################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import torch
from torch import nn
from torch_geometric.utils import scatter

# Typing
from torch import Tensor
from typing import List, Tuple, Optional

####################### Basic modules #######################

class MLP(nn.Module):
    def __init__(self, dims: List[int], act=None) -> None:
        super().__init__()
        self.dims = dims
        self.act = act
        
        num_layers = len(dims)

        layers = []
        for i in range(num_layers-1):
            layers += [nn.Linear(dims[i], dims[i+1])]
            if (act is not None) and (i < num_layers-2):
                layers += [act]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(dims={self.dims}, act={self.act})'

def graph_softmax(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    src_max = scatter(src, index=index, dim=0, dim_size=dim_size, reduce='max')
    out = src - src_max[index]
    out = out.exp()
    out_sum = scatter(out, index=index, dim=0, dim_size=dim_size, reduce='sum') + 1e-12
    out_sum = out_sum[index]
    return out / out_sum

####################### Graphormer convolution modules #######################

import math

class NodeAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, edge_dim: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.dim       = dim
        self.num_heads = num_heads
        self.edge_dim  = edge_dim
        self.lin_Q = nn.Linear(dim, dim)
        self.lin_K = nn.Linear(dim, dim)
        self.lin_V = nn.Linear(dim, dim)
        self.lin_O = nn.Linear(dim, dim)
        self.edge_bias = MLP([edge_dim, edge_dim, num_heads], act=nn.SiLU())
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        num_nodes = x.size(0)
        i = edge_index[0]
        j = edge_index[1]

        # Query, key, and value embedding
        d_k = self.dim // self.num_heads
        query = self.lin_Q(x).view(-1, self.num_heads, d_k)[j]
        key   = self.lin_K(x).view(-1, self.num_heads, d_k)[i]
        value = self.lin_V(x).view(-1, self.num_heads, d_k)[i]

        # Multi-head attention
        scores = (query * key).sum(dim=-1) / math.sqrt(d_k) + self.edge_bias(edge_attr)
        alpha = graph_softmax(scores, index=j, dim_size=num_nodes)
        attn_out = (alpha[..., None] * value).view(-1, self.dim)
        attn_out = scatter(attn_out, index=j, dim=0, dim_size=num_nodes)
        return self.lin_O(attn_out)

class GraphormerConv(nn.Module):
    """Graphormer convolution layer.

    Reference: https://arxiv.org/abs/2306.05445
    """
    def __init__(self, dim: int, num_heads: int, ff_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ff_dim    = ff_dim
        self.edge_dim  = edge_dim
        self.node_attn = NodeAttention(dim, num_heads, edge_dim)
        self.norm1     = nn.LayerNorm(dim)
        self.norm2     = nn.LayerNorm(dim)
        self.ffn       = MLP([dim, ff_dim, dim], act=nn.SiLU())
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        out = self.norm1(x   + self.node_attn(x, edge_index, edge_attr))
        out = self.norm2(out + self.ffn(out))
        return out
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}, ff_dim={self.ff_dim}, edge_dim={self.edge_dim}'

####################### Graphormer high-level modules #######################

import copy

class Encoder(nn.Module):
    def __init__(self, num_species: int, node_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.num_species = num_species
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.embed_node = nn.Embedding(num_species, node_dim)
        self.embed_edge = MLP([1, edge_dim, edge_dim], act=nn.SiLU())
    
    def forward(self, x: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        h_node = self.embed_node(x)
        h_edge = self.embed_edge(edge_attr)
        return h_node, h_edge

class Processor(nn.Module):
    def __init__(self, num_convs: int, node_dim: int, num_heads: int, ff_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.num_convs = num_convs
        self.node_dim  = node_dim
        self.ff_dim    = ff_dim
        self.edge_dim  = edge_dim
        self.convs = nn.ModuleList(
            [copy.deepcopy(GraphormerConv(node_dim, num_heads, ff_dim, edge_dim)) for _ in range(num_convs)]
        )

    def forward(self, h_node: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        for conv in self.convs:
            h_node = conv(h_node, edge_index, edge_attr)
        return h_node

class Graphormer(nn.Module):
    def __init__(self, encoder, processor, global_pool, decoder):
        super().__init__()
        self.encoder     = encoder
        self.processor   = processor
        self.global_pool = global_pool
        self.decoder     = decoder
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        h_node, h_edge = self.encoder(x, edge_attr)
        h_node         = self.processor(h_node, edge_index, h_edge)
        h_pooled       = self.global_pool(h_node, batch=batch)
        return self.decoder(h_pooled)

####################### LightningModule #######################

import lightning as L
from torch_geometric.nn import global_add_pool, global_mean_pool

class LitGraphormer_3d(L.LightningModule):
    def __init__(self, num_species: int, node_dim: int, ff_dim: int, edge_dim: int, num_convs: int, num_heads: int, sum_pooling: bool = True, ema_decay: float = 0.9999, learn_rate: float = 1e-4) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # Core model
        self.model = Graphormer(
            encoder     = Encoder(num_species=num_species, node_dim=node_dim, edge_dim=edge_dim),
            processor   = Processor(num_convs=num_convs, node_dim=node_dim, num_heads=num_heads, ff_dim=ff_dim, edge_dim=edge_dim),
            global_pool = global_add_pool if sum_pooling == True else global_mean_pool,
            decoder     = MLP([node_dim, node_dim, 1], act=nn.SiLU())
        )
        
        # EMA model
        ema_avg = lambda avg_params, params, num_avg: ema_decay*avg_params + (1-ema_decay)*params
        self.ema_model = torch.optim.swa_utils.AveragedModel(self.model, avg_fn=ema_avg)

        # Training parameters
        self.learn_rate = learn_rate

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        return self.model(x, edge_index, edge_attr, batch)

    def _get_loss(self, batch):
        pred_y = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        return torch.nn.functional.mse_loss(pred_y, batch.y)

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True,  batch_size=batch.num_graphs, sync_dist=True)
        self.log('hp_metric',  loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch.num_graphs, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True,  batch_size=batch.num_graphs)
        self.log('hp_metric', loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch.num_graphs)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema_model.update_parameters(self.model)
