################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from typing import Tuple, Union
from torch import nn
from torch.nn import functional as F
import torch
import math


class GCL(nn.Module):
    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        normalization_factor,
        aggregation_method,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.SiLU(),
        attention=False,
    ):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):

        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=2)
        else:
            out = torch.cat([source, target, edge_attr], dim=2)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = split_indices(edge_index)

        batch_size = row.size(0)
        num_edges = row.size(1)
        num_nodes = x.size(1)
        feature_dim = edge_attr.size(2)

        row_indices = (
            (
                torch.zeros((batch_size, num_edges)).to(row.device)
                + (torch.arange(batch_size) * num_nodes)
                .reshape(batch_size, 1)
                .to(row.device)
                + row
            )
            .flatten()
            .long()
        )

        edge_attr = edge_attr.reshape(-1, feature_dim)
        results = x.new_full((batch_size * num_nodes, feature_dim), 0)
        results.index_add_(0, row_indices, edge_attr)
        agg = results.reshape(batch_size, num_nodes, feature_dim)

        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=2)
        else:
            agg = torch.cat([x, agg], dim=2)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(
        self,
        h,
        edge_index,
        edge_attr=None,
        node_attr=None,
        node_mask=None,
        edge_mask=None,
    ):
        row, col = split_indices(edge_index)
        batch_size = row.size(0)
        num_edges = row.size(1)
        num_nodes = h.size(1)
        feature_dim = h.size(2)

        row_indices = (
            (
                torch.zeros((batch_size, num_edges)).to(row.device)
                + (torch.arange(batch_size) * num_nodes)
                .reshape(batch_size, 1)
                .to(row.device)
                + row
            )
            .flatten()
            .long()
        )
        col_indices = (
            (
                torch.zeros((batch_size, num_edges)).to(col.device)
                + (torch.arange(batch_size) * num_nodes)
                .reshape(batch_size, 1)
                .to(col.device)
                + col
            )
            .flatten()
            .long()
        ).to(h.device)

        sources_features = torch.index_select(
            h.reshape(-1, feature_dim), 0, row_indices
        )
        target_features = torch.index_select(h.reshape(-1, feature_dim), 0, col_indices)

        sources_features = sources_features.reshape(batch_size, -1, feature_dim)
        target_features = target_features.reshape(batch_size, -1, feature_dim)

        sources_features = sources_features * edge_mask
        target_features = target_features * edge_mask

        edge_feat, mij = self.edge_model(
            sources_features, target_features, edge_attr, edge_mask
        )
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class InvariantBlock(nn.Module):
    def __init__(
        self,
        hidden_nf,
        edge_feat_nf=2,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=2,
        attention=False,
        norm_constant=1,
        sin_embedding=None,
        normalization_factor=100,
        aggregation_method="sum",
        dropout=0.0,
    ) -> None:
        super(InvariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.dropout = dropout
        gcls = [
            GCL(
                self.hidden_nf,
                self.hidden_nf,
                self.hidden_nf,
                edges_in_d=self.hidden_nf,
                act_fn=act_fn,
                attention=attention,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method,
            )
            for i in range(n_layers)
        ]
        self.gcls = nn.ModuleList(gcls)

        norms = [nn.LayerNorm(self.hidden_nf) for i in range(n_layers)]

        self.norms = nn.ModuleList(norms)

        self.messgae_proj = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, self.hidden_nf),
        )
        self.message_norm = nn.LayerNorm(hidden_nf)
        self.to(self.device)

    def invariant_update(self, h, distances, node_mask=None, edge_mask=None):
        if node_mask is not None:
            h = h * node_mask
            distance_mask = node_mask * node_mask.transpose(1, 2)
            distances = distances * distance_mask
        # breakpoint()
        return F.dropout(h, p=self.dropout, training=self.training) + self.message_norm(
            torch.bmm(
                distances,
                F.dropout(self.messgae_proj(h), p=self.dropout, training=self.training),
            )
        )

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        delta_pos = x.unsqueeze(1) - x.unsqueeze(2)
        distances = torch.norm(delta_pos, dim=-1)
        distances = distances / (distances.sum(-1, keepdim=True) + 1)
        # distances = self.distance_proj(distances.unsqueeze(3)).squeeze(3)

        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        h_prev = h
        for norm, layer in zip(self.norms, self.gcls):
            h, _ = layer(
                h_prev,
                edge_index,
                edge_attr=edge_attr,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
            h += self.invariant_update(h_prev, distances, node_mask, edge_mask)
            h = norm(h) * node_mask
            h_prev = h

        # Important, the bias of the last linear might be non-zero
        # So zero out the padded node features
        if node_mask is not None:
            h = h * node_mask
        return h


class EquivariantBlock(InvariantBlock):
    def __init__(
        self,
        hidden_nf,
        edge_feat_nf=2,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=2,
        attention=True,
        tanh=False,
        norm_constant=1,
        sin_embedding=None,
        normalization_factor=100,
        aggregation_method="sum",
    ):
        super(EquivariantBlock, self).__init__(
            hidden_nf,
            edge_feat_nf=edge_feat_nf,
            device=device,
            act_fn=act_fn,
            n_layers=n_layers,
            attention=attention,
            norm_constant=norm_constant,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
        )

        self.to(self.device)

    def equivariant_update(self, h, x, edge_index, node_mask=None):
        radial, coord_diff = coord2diff(x, edge_index, norm_constant=self.norm_constant)
        normalized_coord_diff = coord_diff / (radial + 1e-8)
        if node_mask is not None:
            h = h * node_mask
            x = torch.scatter_add(
                x,
                1,
                edge_index[:, :, 0].long().unsqueeze(-1).repeat(1, 1, 3),
                normalized_coord_diff,
            )
            x = x * node_mask

        return x

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        h_prev = h
        h = super().forward(h_prev, x, edge_index, node_mask, edge_mask, edge_attr)

        x = self.equivariant_update(h_prev, x, edge_index, node_mask)

        return h, x


class TopologicalGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        aggregation_method="sum",
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        attention=False,
        normalization_factor=1,
        out_node_nf=None,
        include_charge=False,
        reduce="sum",
    ):
        super(TopologicalGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.edge_embedding = nn.Linear(in_edge_nf, self.hidden_nf)

        self.reduce = reduce
        # self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        self.GCLs = nn.ModuleList(
            [
                GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    normalization_factor=normalization_factor,
                    aggregation_method=aggregation_method,
                    edges_in_d=self.hidden_nf,
                    act_fn=act_fn,
                    attention=attention,
                )
                for i in range(n_layers)
            ]
        )

        self.node_dec = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, self.hidden_nf),
        )

        self.graph_dec = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            nn.Softsign(),
            nn.Linear(self.hidden_nf, 1),
        )

        self.include_charges = include_charge
        if self.include_charges:
            self.charge_enc = nn.Sequential(
                nn.Linear(1, self.hidden_nf),
                act_fn,
                nn.Linear(self.hidden_nf, self.hidden_nf),
            )

        self.to(self.device)

    def prop_predict(self, h, node_mask):
        """
        Predicts the output based on the input features and node mask.

        Args:
            h (torch.Tensor): Input features.
            node_mask (torch.Tensor): Node mask.

        Returns:
            torch.Tensor: Predicted output.
        """
        # h = self.embedding_out(h)

        pred = self.graph_dec(h)
        if node_mask is not None:
            pred = pred * node_mask

        if self.reduce == "max":
            # if self.training is False:
            # breakpoint()s
            pred, _ = torch.max(h, dim=1)
            return pred

        pred = torch.sum(pred, dim=1)

        if self.reduce == "mean":
            pred = pred / node_mask.sum(1)
        return pred

    def check_input(self, h, edges, charges, edge_attr, node_mask, edge_mask):
        """
        Checks the validity of the input tensors for the GNN model.

        Args:
            h (torch.Tensor): Tensor representing the node features.
            edges (torch.Tensor): Tensor representing the edges in the graph.
            charges (torch.Tensor): Tensor representing the charges of the nodes.
            edge_attr (torch.Tensor): Tensor representing the edge attributes.
            node_mask (torch.Tensor): Tensor representing the node mask.
            edge_mask (torch.Tensor): Tensor representing the edge mask.

        Raises:
            AssertionError: If the input tensors do not meet the specified conditions.

        Returns:
            None
        """
        batch_size = h.size(0)
        num_nodes = h.size(1)
        num_edges = edges.size(1)

        assert (
            h.size(2) == self.embedding.in_features
        ), f" Expected atom features ({h.size(2)}) to be {self.embedding.in_features}"
        assert batch_size == edges.size(0)

        if edge_attr is not None:
            assert (
                edge_attr.size(0) == batch_size
            ), f"{edge_attr.size(0)} != {batch_size}"
            assert edge_attr.size(1) == num_edges, f"{edge_attr.size(1)} != {num_edges}"
            assert (
                edge_attr.size(2) == self.edge_embedding.in_features
            ), f"{edge_attr.size(2)} != {self.edge_embedding.in_features}"
        if node_mask is not None:
            assert node_mask.size() == (batch_size, num_nodes, 1)

        if self.include_charges:
            if charges is None:
                raise ValueError("Charges are required")

        if node_mask is not None:
            assert node_mask.size() == (batch_size, num_nodes, 1)

    def process_node_features(self, h, charges, node_mask):
        """
        Process the node features by applying embedding, charge encoding, and node mask.

        Args:
            h (torch.Tensor): The input node features.
            charges (torch.Tensor): The charges associated with the nodes.
            node_mask (torch.Tensor): The mask indicating which nodes to include.

        Returns:
            torch.Tensor: The processed node features.

        """
        h = self.embedding(h)
        if charges is not None:
            charges = self.charge_enc(charges)
            h = h + charges
        if node_mask is not None:
            h = h * node_mask
        return h

    def process_edge_features(self, edge_attr, edge_mask):
        if edge_attr is not None:
            edge_attr = self.edge_embedding(edge_attr)
            if edge_mask is not None:
                edge_attr = edge_attr * edge_mask
        return edge_attr

    def forward(
        self, h, edges, edge_attr=None, node_mask=None, edge_mask=None, charges=None
    ):
        self.check_input(h, edges, charges, edge_attr, node_mask, edge_mask)
        h = self.process_node_features(h, charges, node_mask)
        edge_attr = self.process_edge_features(edge_attr, edge_mask)
        for i in range(0, self.n_layers):
            h, _ = self.GCLs[i](
                h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask
            )

        prop = self.prop_predict(h, node_mask)
        return prop


class InvariantGNN(TopologicalGNN):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        aggregation_method="sum",
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        attention=False,
        normalization_factor=1,
        out_node_nf=None,
        sin_embedding=False,
        norm_constant=1,
        inv_sublayers=1,
        include_charge=False,
        reduce="sum",
    ) -> None:
        super(InvariantGNN, self).__init__(
            in_node_nf,
            in_edge_nf,
            hidden_nf,
            aggregation_method=aggregation_method,
            device=device,
            act_fn=act_fn,
            n_layers=n_layers,
            attention=attention,
            normalization_factor=normalization_factor,
            out_node_nf=out_node_nf,
            include_charge=include_charge,
            reduce=reduce,
        )

        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        block_list = [
            InvariantBlock(
                hidden_nf,
                edge_feat_nf=edge_feat_nf,
                device=device,
                act_fn=act_fn,
                n_layers=inv_sublayers,
                attention=attention,
                norm_constant=norm_constant,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method,
            )
            for i in range(n_layers)
        ]
        self.GCLs = nn.ModuleList(block_list)

        self.to(self.device)

    def forward(
        self, h, x, edges, edge_attr=None, charges=None, node_mask=None, edge_mask=None
    ):
        # Edit Emiel: Remove velocity as input
        self.check_input(h, edges, charges, edge_attr, node_mask, edge_mask)
        h = self.process_node_features(h, charges, node_mask)
        edge_attr = self.process_edge_features(edge_attr, edge_mask)
        for layer in self.GCLs:
            h = layer(
                h,
                x,
                edges,
                node_mask=node_mask,
                edge_mask=edge_mask,
                edge_attr=edge_attr,
            )

        prop = self.prop_predict(h, node_mask)
        return prop


class EquivariantGNN(InvariantGNN):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=3,
        attention=False,
        norm_diff=True,
        out_node_nf=None,
        tanh=False,
        coords_range=15,
        norm_constant=1,
        inv_sublayers=2,
        sin_embedding=False,
        include_charge=False,
        normalization_factor=100,
        aggregation_method="sum",
        reduce="sum",
    ):
        super(EquivariantGNN, self).__init__(
            in_node_nf,
            in_edge_nf,
            hidden_nf,
            aggregation_method=aggregation_method,
            device=device,
            act_fn=act_fn,
            n_layers=n_layers,
            attention=attention,
            normalization_factor=normalization_factor,
            out_node_nf=out_node_nf,
            sin_embedding=sin_embedding,
            include_charge=include_charge,
            reduce=reduce,
        )

        self.coords_range_layer = (
            float(coords_range / n_layers) if n_layers > 0 else float(coords_range)
        )

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        equivariant_blocks = [
            EquivariantBlock(
                hidden_nf,
                edge_feat_nf=edge_feat_nf,
                device=device,
                act_fn=act_fn,
                n_layers=inv_sublayers,
                attention=attention,
                tanh=tanh,
                norm_constant=norm_constant,
                sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method,
            )
            for i in range(n_layers)
        ]
        self.GCLs = nn.ModuleList(equivariant_blocks)
        self.to(self.device)

    def forward(
        self, h, x, edges, edge_attr, charges=None, node_mask=None, edge_mask=None
    ):
        # Edit Emiel: Remove velocity as input
        # distances, _ = coord2diff(x, edges)
        # if self.sin_embedding is not None:
        #     distances = self.sin_embedding(distances)
        self.check_input(h, edges, charges, edge_attr, node_mask, edge_mask)
        h = self.process_node_features(h, charges, node_mask)
        edge_attr = self.process_edge_features(edge_attr, edge_mask)

        for layer in self.GCLs:
            h, x = layer(
                h,
                x,
                edges,
                node_mask=node_mask,
                edge_mask=edge_mask,
                edge_attr=edge_attr,
            )

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        prop = self.prop_predict(h, node_mask)

        # breakpoint()
        return prop


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15.0, min_res=15.0 / 2000.0, div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = (
            2 * math.pi * div_factor ** torch.arange(self.n_frequencies) / max_res
        )
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1) -> Tuple[torch.Tensor, torch.Tensor]:
    row, col = split_indices(edge_index)
    row = row.long().unsqueeze(-1).repeat(1, 1, x.size(2))
    col = col.long().unsqueeze(-1).repeat(1, 1, x.size(2))
    coord_diff = x.gather(1, row) - x.gather(1, col)
    radial = torch.sum((coord_diff) ** 2, -1).unsqueeze(-1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(
    data, segment_ids, num_segments, normalization_factor, aggregation_method: str
):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == "sum":
        result = result / normalization_factor

    if aggregation_method == "mean":
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


def split_indices(index):
    row = index[:, :, 0]
    col = index[:, :, 1]
    return row, col


def segment_sum(data, indices, max_nodes, normalization_factor, aggregation_method):
    results_shape = (data.size(0), max_nodes, data.size(2))
    results = data.new_full(results_shape, 0)
    results.scatter_add_(1, indices, data)
    if aggregation_method == "sum":
        results = results / normalization_factor
    else:
        raise ValueError("Invalid aggregation")
    return results


if __name__ == "__main__":
    # Define model
    # model = TopologicalGNN(in_node_nf=4, in_edge_nf=4, hidden_nf=64, device="cpu")
    # print(model)

    # from dataset import TorchMolDataset, batched_mol_collator

    # dataset = TorchMolDataset(property="density", split="train")
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=4, collate_fn=batched_mol_collator
    # )

    # for batch in dataloader:
    #     (
    #         batch_atom_features,
    #         batch_atom_charges,
    #         batch_bond_features,
    #         batch_bond_index,
    #         batch_property,
    #         batch_num_atoms,
    #         node_mask,
    #         edge_mask,
    #     ) = batch
    #     print(batch_atom_features.shape)
    #     print(batch_atom_charges.shape)
    #     print(batch_bond_features.shape)
    #     print(batch_bond_index.shape)
    #     print(batch_property.shape)
    #     print(batch_num_atoms.shape)
    #     print(node_mask.shape)
    #     print(edge_mask.shape)

    #     # Forward pass
    #     prop = model(
    #         batch_atom_features,
    #         batch_bond_index,
    #         edge_attr=batch_bond_features,
    #         node_mask=node_mask,
    #         edge_mask=edge_mask,
    #         charges=batch_atom_charges,
    #     )
    #     break

    from dataset import Torch3DDataset, batched_3D_collator

    # model = InvariantGNN(
    #     in_node_nf=5, in_edge_nf=4, hidden_nf=64, include_charge=True, device="cpu"
    # )

    model = EquivariantGNN(
        in_node_nf=5, in_edge_nf=4, hidden_nf=64, include_charge=True, device="cpu"
    )

    dataset = Torch3DDataset(property="density", split="train")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, collate_fn=batched_3D_collator
    )

    for batch in dataloader:
        h = batch["atom_features"]
        x = batch["coords"]
        charges = batch["atom_charges"]
        edge_index = batch["bond_index"]
        node_mask = batch["node_mask"]
        edge_mask = batch["edge_mask"]
        edge_attr = batch["bond_features"]
        prop = model(
            h,
            x,
            edge_index,
            edge_attr=edge_attr,
            node_mask=node_mask,
            edge_mask=edge_mask,
            charges=charges,
        )
        break
