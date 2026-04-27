
import torch
from torch.optim.lr_scheduler import StepLR
from ..nn.gears_modules import GEARS_Model,loss_fct,uncertainty_loss_fct
from .base import  PerturbationModel
import numpy as np
from VCBench.data.utils import GraphBuilder
import warnings
import networkx as nx
import pickle

torch.manual_seed(0)

warnings.filterwarnings("ignore")


class GEARS(PerturbationModel):
    """
    GEARS base model class
    """

    def __init__(self,
                 datamodule,
                 data_path,
                 pert_key,
                 comb_delim,
                 control_val,
                 gene2go_path,
                 build_GO_workers,
                 gene_set_path,  # Required: must provide external pkl file
                 seed=42,
                 use_mask: bool = False,  # Unified mask switch for training loss and evaluation
                 use_covs: bool = False,  # Unified covariate usage parameter
                 lr=1e-3,
                 wd=5e-4,
                 lr_scheduler_freq: float | None = None,
                 lr_scheduler_interval: str | None = None,
                 lr_scheduler_patience: float | None = None,
                 lr_scheduler_factor: float | None = None,
                 lr_monitor_key: str | None = None,
                 lr_scheduler_mode: str | None = None,
                 lr_scheduler_max_lr: float | None = None,
                 lr_scheduler_total_steps: int | None = None,
                 **kwargs):
        super(GEARS,self).__init__(datamodule,
                                   lr=lr,
                                   wd=wd,
                                   lr_scheduler_freq=lr_scheduler_freq,
                                   lr_scheduler_interval=lr_scheduler_interval,
                                   lr_scheduler_patience=lr_scheduler_patience,
                                   lr_scheduler_factor=lr_scheduler_factor,
                                   lr_monitor_key=lr_monitor_key,
                                   lr_scheduler_mode=lr_scheduler_mode,
                                   lr_scheduler_max_lr=lr_scheduler_max_lr,
                                   lr_scheduler_total_steps=lr_scheduler_total_steps,
                                   use_mask=use_mask,  # Pass use_mask to base class
                                   )

        # Auto-configure covariate usage based on data transform's use_covs setting or parameter
        if hasattr(datamodule.train_dataset.transform, 'use_covs') and datamodule.train_dataset.transform.use_covs:
            # If data transform enables covariates, automatically enable covariate injection
            use_covs = True

        self.use_covs = use_covs
        self.pert_key=pert_key
        self.comb_delim=comb_delim
        self.control_val=control_val
        self.data_path=data_path
        self.seed = seed

        self.config = None
        adata = datamodule.adata
        
        # self.deg_genes=adata.uns['deg_genes']
        # deg_idxs=[]

        # Some preprocess pipelines store a DEG gene list in adata.uns['deg_genes'].
        # Fallback gracefully if it's missing by using all genes.
        if 'deg_genes' in adata.uns:
            self.deg_genes = adata.uns['deg_genes']
        else:
            # Fallback: use all genes as "deg_genes"
            self.deg_genes = list(self.gene_names)

        deg_idxs = []
        for g in self.deg_genes:
            deg_idxs.append((self.gene_names == g).argmax())
        self.deg_idxs = np.array(deg_idxs)

        self.control_adata=adata[adata.obs[self.pert_key]==self.control_val]

        self.dict_filter=self.get_dropout_non_zero_genes(adata)

        # Force use of external pkl file - gene_set_path is required
        if gene_set_path is None:
            raise ValueError("gene_set_path must be provided. External pkl file is required.")
        
        # Load perturbation list from external pkl file
        with open(gene_set_path, 'rb') as f:
            pert_list = pickle.load(f)
        
        # Fallback logic commented out - must use external pkl
        # else:
        #     pert_list = self.get_perts_list(adata)

        # Load gene2go mapping from external pkl file (required)
        with open( gene2go_path, 'rb') as f:
            gene2go = pickle.load(f)

        self.gene2go = {i: gene2go[i] for i in pert_list if i in gene2go.keys()}
        self.pert_list = list(self.gene2go.keys())
        self.node_map_pert = {x: it for it, x in enumerate(self.pert_list)}
        self.n_perts = len(self.pert_list)
        self.node_map={x:it for it, x in enumerate(self.gene_names)}
        self.get_ctrl_expr()

        self.g_builder = GraphBuilder(data_dir=self.data_path,
                                      pert_key=self.pert_key,
                                      control_val=self.control_val,
                                      comb_delim=self.comb_delim,
                                      num_workers=build_GO_workers)
        self.model_initialize(**kwargs)



    def get_dropout_non_zero_genes(self,adata):

        non_zeros_gene_idx = {}
        for pert in adata.obs[self.pert_key].unique():
            X = np.mean(adata[adata.obs[self.pert_key] == pert].X, axis=0)
            non_zero = np.where(np.array(X) != 0)[0]
            non_zeros_gene_idx[pert] = np.sort(non_zero)

        return non_zeros_gene_idx

    def get_ctrl_expr(self):
        ctrl_expression_mean = torch.tensor(
            np.mean(self.control_adata.X,
                    axis=0)).reshape(-1, )
        self.register_buffer('ctrl_expression_mean', ctrl_expression_mean)

    def get_perts_list(self,adata):
        adata=adata[adata.obs[self.pert_key] != self.control_val]
        perts_list = []
        for comb_pert in adata.obs[self.pert_key].unique():
            perts_list.extend(comb_pert.split(self.comb_delim))
        perts_list.extend(list(self.gene_names))
        return list(set(perts_list))

    def model_initialize(self,
                         hidden_size=64,
                         num_go_gnn_layers=1,
                         num_gene_gnn_layers=1,
                         decoder_hidden_size=16,
                         num_similar_genes_go_graph=20,
                         num_similar_genes_co_express_graph=20,
                         coexpress_threshold=0.4,
                         uncertainty=False,
                         uncertainty_reg=1,
                         direction_lambda=1e-1,
                         G_go=None,
                         G_go_weight=None,
                         G_coexpress=None,
                         G_coexpress_weight=None,
                         no_perturb=False,
                         **kwargs
                         ):
        """
        Initialize the model

        Parameters
        ----------
        hidden_size: int
            hidden dimension, default 64
        num_go_gnn_layers: int
            number of GNN layers for GO graph, default 1
        num_gene_gnn_layers: int
            number of GNN layers for co-expression gene graph, default 1
        decoder_hidden_size: int
            hidden dimension for gene-specific decoder, default 16
        num_similar_genes_go_graph: int
            number of maximum similar K genes in the GO graph, default 20
        num_similar_genes_co_express_graph: int
            number of maximum similar K genes in the co expression graph, default 20
        coexpress_threshold: float
            pearson correlation threshold when constructing coexpression graph, default 0.4
        uncertainty: bool
            whether or not to turn on uncertainty mode, default False
        uncertainty_reg: float
            regularization term to balance uncertainty loss and prediction loss, default 1
        direction_lambda: float
            regularization term to balance direction loss and prediction loss, default 1
        G_go: scipy.sparse.csr_matrix
            GO graph, default None
        G_go_weight: scipy.sparse.csr_matrix
            GO graph edge weights, default None
        G_coexpress: scipy.sparse.csr_matrix
            co-expression graph, default None
        G_coexpress_weight: scipy.sparse.csr_matrix
            co-expression graph edge weights, default None
        no_perturb: bool
            predict no perturbation condition, default False

        Returns
        -------
        None
        """
        self.config = {'hidden_size': hidden_size,
                       'num_go_gnn_layers': num_go_gnn_layers,
                       'num_gene_gnn_layers': num_gene_gnn_layers,
                       'decoder_hidden_size': decoder_hidden_size,
                       'num_similar_genes_go_graph': num_similar_genes_go_graph,
                       'num_similar_genes_co_express_graph': num_similar_genes_co_express_graph,
                       'coexpress_threshold': coexpress_threshold,
                       'uncertainty': uncertainty,
                       'uncertainty_reg': uncertainty_reg,
                       'direction_lambda': direction_lambda,
                       'G_go': G_go,
                       'G_go_weight': G_go_weight,
                       'G_coexpress': G_coexpress,
                       'G_coexpress_weight': G_coexpress_weight,
                       'device': self.device,
                       'num_genes': self.n_genes,
                       'num_perts': self.n_perts,
                       'no_perturb': no_perturb
                       }


        if self.config['G_coexpress'] is None:
            ## calculating co expression similarity graph
            edge_list = self.g_builder.get_similarity_network(network_type='co-express',
                                                              control_adata=self.control_adata,
                                                              k=num_similar_genes_co_express_graph,
                                                              threshold=coexpress_threshold,
                                                              gene2go=self.gene2go,
                                                              )

            sim_network = GeneSimNetwork(edge_list, self.gene_names, node_map=self.node_map)
            self.config['G_coexpress'] = sim_network.edge_index
            self.config['G_coexpress_weight'] = sim_network.edge_weight

        if self.config['G_go'] is None:
            ## calculating gene ontology similarity graph
            edge_list = self.g_builder.get_similarity_network(network_type='go',
                                                              control_adata=self.control_adata,
                                                              k=num_similar_genes_go_graph,
                                                              threshold=0.1,
                                                              gene2go=self.gene2go,
                                                              )

            sim_network = GeneSimNetwork(edge_list, self.pert_list, node_map=self.node_map_pert)
            self.config['G_go'] = sim_network.edge_index
            self.config['G_go_weight'] = sim_network.edge_weight

        self.model = GEARS_Model(self.config)

        # Add covariate projection layer if covariates are enabled
        if self.use_covs and hasattr(self, 'cov_dims') and self.cov_dims:
            cov_dim = sum(self.cov_dims.values())
            hidden_size = self.config['hidden_size']
            # Project covariates to embedding dimension
            self.model.cov_proj = torch.nn.Linear(hidden_size + cov_dim, hidden_size)


    def perts2idxslist(self,perts):
        pert_idxs_list=[]
        for comb_pert in perts:
            pert_idxs=[]
            for pert in comb_pert.split(self.comb_delim):
                if pert==self.control_val:
                    pert_idx=-1
                else:
                    pert_idx=self.node_map_pert.get(pert,None)
                    if pert_idx is None:
                        pert_=np.random.choice(self.pert_list,size=1)[0]
                        pert_idx=self.node_map_pert[pert_]
                pert_idxs.append(pert_idx)
            pert_idxs_list.append(pert_idxs)
        return pert_idxs_list

    def get_input_dict(self,batch):
        x=batch['control_cell_counts']
        perts=batch[self.pert_key]
        pert_idx=self.perts2idxslist(perts)

        input_dict = {
            'x': x,
            'pert_idx': pert_idx,
        }

        # Add covariates if enabled
        if self.use_covs and hasattr(self, 'cov_keys') and self.cov_keys:
            covariates = {cov_key: batch[cov_key] for cov_key in self.cov_keys if cov_key in batch}
            if covariates:
                # Concatenate all covariate embeddings
                cov_tensors = [cov for cov in covariates.values() if cov is not None and len(cov) > 0]
                if cov_tensors:
                    merged_covariates = torch.cat(cov_tensors, dim=-1)
                    input_dict['covariates'] = merged_covariates

        return input_dict

    def training_step(self, batch, batch_idx):
        input_dict = self.get_input_dict(batch)
        y=batch['pert_cell_counts']

        # Get expression mask using unified method from base class
        mask = self._get_mask(batch)
        if mask is not None:
            mask = mask.to(y.device)

        if self.config['uncertainty']:
            pred, logvar = self.model(input_dict)
            loss = uncertainty_loss_fct(pred, logvar, y, batch[self.pert_key],
                                        reg=self.config['uncertainty_reg'],
                                        ctrl=self.ctrl_expression_mean,
                                        dict_filter=self.dict_filter,
                                        direction_lambda=self.config['direction_lambda'],
                                        control_val=self.control_val,
                                        mask=mask)
        else:
            pred = self.model(input_dict)
            loss = loss_fct(pred, y, batch[self.pert_key],
                            ctrl=self.ctrl_expression_mean,
                            dict_filter=self.dict_filter,
                            direction_lambda=self.config['direction_lambda'],
                            control_val=self.control_val,
                            mask=mask)

        self.log("train_loss", loss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True,
                 batch_size=y.shape[0])

        return loss

    def validation_step(self, data_tuple, batch_idx):
        batch,_=data_tuple
        pred=self.predict(batch)
        # Compatible with both dict and Batch objects
        y = batch['pert_cell_counts'] if isinstance(batch, dict) else batch.pert_cell_counts

        # Get expression mask using unified method from base class
        mask = self._get_mask(batch)
        if mask is not None:
            mask = mask.to(y.device)

        # Simple masked MSE for validation
        mse = (pred - y) ** 2
        if mask is not None:
            valid = mask.sum(dim=1)
            loss_per_batch = (mse * mask).sum(dim=1)
            loss = (loss_per_batch / valid).nanmean()
        else:
            loss = torch.mean(mse, dim=-1).mean()

        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=y.shape[0], on_step=True, on_epoch=True)

        return loss

    def predict(self, batch):
        input_dict = self.get_input_dict(batch)

        if self.config['uncertainty']:
            pred, _ = self.model(input_dict)
        else:
            pred = self.model(input_dict)
        return pred

    def configure_optimizers(self):
        """
        Configure optimizer and scheduler for GEARS.
        Supports multiple scheduler modes: onecycle, plateau, or step (default).
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )

        if self.lr_scheduler_mode == "onecycle":
            # OneCycleLR scheduler
            total_steps = self.lr_scheduler_total_steps
            if total_steps is None:
                try:
                    steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
                    total_steps = steps_per_epoch * self.trainer.max_epochs
                except Exception:
                    total_steps = 100 * 100  # fallback
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr_scheduler_max_lr or self.lr,
                total_steps=total_steps,
            )
            lr_scheduler = {"scheduler": scheduler, "interval": "step"}

        elif self.lr_scheduler_mode == "plateau":
            # ReduceLROnPlateau scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": self.lr_monitor_key,
                "frequency": self.lr_scheduler_freq,
                "interval": self.lr_scheduler_interval,
            }

        else:
            # Default: StepLR (GEARS original implementation with step_size=1, gamma=0.5)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": self.lr_monitor_key,
                "frequency": self.lr_scheduler_freq,
                "interval": self.lr_scheduler_interval,
            }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class GeneSimNetwork():
    """
    GeneSimNetwork class

    Args:
        edge_list (pd.DataFrame): edge list of the network
        gene_list (list): list of gene names
        node_map (dict): dictionary mapping gene names to node indices

    Attributes:
        edge_index (torch.Tensor): edge index of the network
        edge_weight (torch.Tensor): edge weight of the network
        G (nx.DiGraph): networkx graph object
    """

    def __init__(self, edge_list, gene_list, node_map):
        """
        Initialize GeneSimNetwork class
        """

        edge_list = edge_list.copy()
        edge_list = edge_list.dropna(subset=['source', 'target'])
        mask = edge_list['source'].isin(node_map.keys()) & edge_list['target'].isin(node_map.keys())
        edge_list = edge_list.loc[mask]
        self.edge_list = edge_list
        self.G = nx.from_pandas_edgelist(
            self.edge_list,
            source='source',
            target='target',
            edge_attr=['importance'],
            create_using=nx.DiGraph(),
        )

        # self.edge_list = edge_list
        # self.G = nx.from_pandas_edgelist(self.edge_list, source='source',
        #                                  target='target', edge_attr=['importance'],
        #                                  create_using=nx.DiGraph())
        self.gene_list = gene_list
        for n in self.gene_list:
            if n not in self.G.nodes():
                self.G.add_node(n)

        edge_index_ = [(node_map[e[0]], node_map[e[1]]) for e in
                       self.G.edges]
        self.edge_index = torch.tensor(edge_index_, dtype=torch.long).T

        edge_attr = nx.get_edge_attributes(self.G, 'importance')
        importance = np.array([edge_attr[e] for e in self.G.edges])
        self.edge_weight = torch.Tensor(importance)


