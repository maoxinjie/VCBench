import torch
import torch.nn as nn
from torch_geometric.nn import SGConv
import numpy as np



class MLP(torch.nn.Module):

    def __init__(self, sizes, norm=True, last_layer_act="linear"):
        """
        Multi-layer perceptron
        :param sizes: list of sizes of the layers
        :param batch_norm: whether to use batch normalization
        :param last_layer_act: activation function of the last layer

        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [j for j in layers if j is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.network(x)


class GEARS_Model(torch.nn.Module):
    """
    GEARS model

    """

    def __init__(self, args):
        """
        :param args: arguments dictionary
        """

        super(GEARS_Model, self).__init__()
        self.args = args
        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        hidden_size = args['hidden_size']
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_go_gnn_layers']
        self.indv_out_hidden_size = args['decoder_hidden_size']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']
        self.no_perturb = args['no_perturb']
        self.pert_emb_lambda = 0.2

        # Covariate projection layer (initialized later if needed)
        self.cov_proj = None

        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)

        # gene/globel perturbation embedding dictionary lookup
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_perts, hidden_size, max_norm=True)

        # transformation layer
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')

        # gene co-expression GNN
        self.register_buffer('G_coexpress', args['G_coexpress'])
        self.register_buffer('G_coexpress_weight', args['G_coexpress_weight'])

        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.layers_emb_pos = torch.nn.ModuleList()
        for i in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))

        ### perturbation gene ontology GNN
        self.register_buffer('G_sim', args['G_go'])
        self.register_buffer('G_sim_weight', args['G_go_weight'])

        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))

        # decoder shared MLP
        self.recovery_w = MLP([hidden_size, hidden_size * 2, hidden_size], last_layer_act='linear')

        # gene specific decoder
        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes,
                                               hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)

        # Cross gene MLP
        self.cross_gene_state = MLP([self.num_genes, hidden_size,
                                     hidden_size])
        # final gene specific decoder
        self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes,
                                               hidden_size + 1))
        self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)

        # batchnorms
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)

        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size * 2, hidden_size, 1], last_layer_act='linear')

        self.register_buffer('gene_range', torch.arange(self.num_genes, dtype=torch.long))
        self.register_buffer('pert_range', torch.arange(self.num_perts, dtype=torch.long))

    def forward(self, data):
        """
        Forward pass of the model
        """

        x, pert_idx = data['x'], data['pert_idx']
        covariates = data.get('covariates', None)
        if self.no_perturb:
            out = x.reshape(-1, 1)
            out = torch.split(torch.flatten(out), self.num_genes)
            return torch.stack(out)
        else:
            self.G_coexpress = self.G_coexpress.to(x.device)
            self.G_coexpress_weight = self.G_coexpress_weight.to(x.device)
            self.G_sim= self.G_sim.to(x.device)
            self.G_sim_weight = self.G_sim_weight.to(x.device)


            num_graphs = x.shape[0]

            ## get base gene embeddings
            emb = self.gene_emb(self.gene_range.repeat(num_graphs, ))
            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb)

            pos_emb = self.emb_pos(self.gene_range.repeat(num_graphs, ))
            for idx, layer in enumerate(self.layers_emb_pos):
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    pos_emb = pos_emb.relu()

            base_emb = base_emb + 0.2 * pos_emb
            base_emb = self.emb_trans_v2(base_emb)

            # Add covariates if provided
            if covariates is not None and hasattr(self, 'cov_proj') and self.cov_proj is not None:
                # Expand covariates to match gene embedding dimensions
                # covariates shape: [batch_size, cov_dim]
                # base_emb shape: [batch_size * num_genes, emb_dim]
                covariates_expanded = covariates.unsqueeze(1).repeat(1, self.num_genes, 1)  # [batch_size, num_genes, cov_dim]
                covariates_flat = covariates_expanded.reshape(-1, covariates.size(-1))  # [batch_size * num_genes, cov_dim]

                # Concatenate covariates with base embeddings
                combined_emb = torch.cat([base_emb, covariates_flat], dim=-1)

                # Project back to original embedding dimension
                base_emb = self.cov_proj(combined_emb)

            ## get perturbation index and embeddings

            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1:
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T

            pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(x.device))

            ## augment global perturbation embedding with GNN
            for idx, layer in enumerate(self.sim_layers):
                pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()

            ## add global perturbation embedding to each gene in each cell in the batch
            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

            if pert_index.shape[0] != 0:
                ### in case all samples in the batch are controls, then there is no indexing for pert_index.
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:
                        # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                    for idx, j in enumerate(pert_track.keys()):
                        base_emb[j] = base_emb[j] + emb_total[idx]

            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
            base_emb = self.bn_pert_base(base_emb)

            ## apply the first MLP
            base_emb = self.transform(base_emb)
            out = self.recovery_w(base_emb)
            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis=2)
            out = w + self.indv_b1

            # Cross gene
            cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
            cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

            cross_gene_embed = cross_gene_embed.reshape([num_graphs, self.num_genes, -1])
            cross_gene_out = torch.cat([out, cross_gene_embed], 2)

            cross_gene_out = cross_gene_out * self.indv_w2
            cross_gene_out = torch.sum(cross_gene_out, axis=2)
            out = cross_gene_out + self.indv_b2
            out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1, 1)
            out = torch.split(torch.flatten(out), self.num_genes)

            ## uncertainty head
            if self.uncertainty:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
                return torch.stack(out), torch.stack(out_logvar)

            return torch.stack(out)


def loss_fct(pred, y, perts, ctrl=None, direction_lambda=1e-3, dict_filter=None, control_val=None, mask=None):
    """
    Main MSE Loss function, includes direction loss

    Args:
        pred (torch.tensor): predicted values
        y (torch.tensor): true values
        perts (list): list of perturbations
        ctrl (str): control perturbation
        direction_lambda (float): direction loss weight hyperparameter
        dict_filter (dict): dictionary of perturbations to conditions
        control_val: control value to identify control samples
        mask (torch.tensor): optional per-sample expression mask [N, G]

    """
    gamma = 2
    perts = np.array(perts)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)

    for p in set(perts):
        pert_idx = np.where(perts == p)[0]

        # during training, we remove the all zero genes into calculation of loss.
        # this gives a cleaner direction loss. empirically, the performance stays the same.
        if p != control_val:
            retain_idx = dict_filter[p]
            pred_p = pred[pert_idx][:, retain_idx]
            y_p = y[pert_idx][:, retain_idx]
            mask_p = mask[pert_idx][:, retain_idx] if mask is not None else None
        else:
            pred_p = pred[pert_idx]
            y_p = y[pert_idx]
            mask_p = mask[pert_idx] if mask is not None else None

        mse = (pred_p - y_p) ** (2 + gamma)

        # Apply mask if available
        if mask_p is not None:
            valid = mask_p.sum()
            if valid > 0:
                losses = losses + (mse * mask_p).sum() / valid
            else:
                losses = losses + mse.mean()
        else:
            losses = losses + torch.sum(mse) / pred_p.shape[0] / pred_p.shape[1]

        ## direction loss (not masked, as it measures direction correctness)
        if (p != control_val):
            losses = losses + torch.sum(direction_lambda *
                                        (torch.sign(y_p - ctrl[retain_idx]) -
                                         torch.sign(pred_p - ctrl[retain_idx])) ** 2) / \
                     pred_p.shape[0] / pred_p.shape[1]
        else:
            losses = losses + torch.sum(direction_lambda * (torch.sign(y_p - ctrl) -
                                                            torch.sign(pred_p - ctrl)) ** 2) / \
                     pred_p.shape[0] / pred_p.shape[1]
    return losses / (len(set(perts)))


def uncertainty_loss_fct(pred, logvar, y, perts, reg=0.1, ctrl=None,
                         direction_lambda=1e-3, dict_filter=None, control_val=None, mask=None):
    """
    Uncertainty loss function

    Args:
        pred (torch.tensor): predicted values
        logvar (torch.tensor): log variance
        y (torch.tensor): true values
        perts (list): list of perturbations
        reg (float): regularization parameter
        ctrl (str): control perturbation
        direction_lambda (float): direction loss weight hyperparameter
        dict_filter (dict): dictionary of perturbations to conditions
        control_val: control value to identify control samples
        mask (torch.tensor): optional per-sample expression mask [N, G]

    """
    gamma = 2
    perts = np.array(perts)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
    for p in set(perts):
        pert_idx = np.where(perts == p)[0]
        if p != control_val:
            retain_idx = dict_filter[p]
            pred_p = pred[pert_idx][:, retain_idx]
            y_p = y[pert_idx][:, retain_idx]
            logvar_p = logvar[pert_idx][:, retain_idx]
            mask_p = mask[pert_idx][:, retain_idx] if mask is not None else None
        else:
            pred_p = pred[pert_idx]
            y_p = y[pert_idx]
            logvar_p = logvar[pert_idx]
            mask_p = mask[pert_idx] if mask is not None else None

        # uncertainty based loss
        mse = (pred_p - y_p) ** (2 + gamma)
        uncertainty_term = reg * torch.exp(-logvar_p) * mse
        total_loss = mse + uncertainty_term

        # Apply mask if available
        if mask_p is not None:
            valid = mask_p.sum()
            if valid > 0:
                losses += (total_loss * mask_p).sum() / valid
            else:
                losses += total_loss.mean()
        else:
            losses += torch.sum(total_loss) / pred_p.shape[0] / pred_p.shape[1]

        # direction loss (not masked, as it measures direction correctness)
        if p != control_val:
            losses += torch.sum(direction_lambda *
                                (torch.sign(y_p - ctrl[retain_idx]) -
                                 torch.sign(pred_p - ctrl[retain_idx])) ** 2) / \
                      pred_p.shape[0] / pred_p.shape[1]
        else:
            losses += torch.sum(direction_lambda *
                                (torch.sign(y_p - ctrl) -
                                 torch.sign(pred_p - ctrl)) ** 2) / \
                      pred_p.shape[0] / pred_p.shape[1]

    return losses / (len(set(perts)))