import lightning as L
import torch
from abc import ABC
import pandas as pd
import numpy as np
import anndata as ad
import os
import gc
from hydra.core.hydra_config import HydraConfig
from ...analysis.benchmarks.evaluation import Evaluation
from lightning_utilities.core.apply_func import apply_to_collection

class Batch:
    def __init__(self,batch_dict):
        self.batch_dict = batch_dict
        for k in batch_dict:
            setattr(self, k, batch_dict[k])
    def __getitem__(self,key):
        return self.batch_dict[key]
    def __len__(self):
        return len(list(self.batch_dict.values())[0])
    def __iter__(self):
        for key in self.batch_dict:
            yield key
    def get(self,key,default=None):
        return self.batch_dict.get(key,default)
    def keys(self):
        return self.batch_dict.keys()
    def items(self):
        return self.batch_dict.items()
    def values(self):
        return self.batch_dict.values()


class PerturbationModel(L.LightningModule, ABC):

    def __init__(
        self,
        datamodule: L.LightningDataModule | None = None,
        lr: float | None = None,
        wd: float | None = None,
        lr_scheduler_freq: float | None = None,
        lr_scheduler_interval: str | None = None,
        lr_scheduler_patience: float | None = None,
        lr_scheduler_factor: float | None = None,
        lr_scheduler_mode: str | None = None,  # "plateau", "onecycle", "step"
        lr_scheduler_max_lr: float | None = None,  # For OneCycleLR
        lr_scheduler_total_steps: int | None = None,  # For OneCycleLR
        lr_monitor_key: str | None = None,
        use_infer_top_hvgs: bool=False,
        use_mask: bool = False,  # Unified mask switch for both training and evaluation
        **kwargs,
    ):
        super(PerturbationModel, self).__init__()

        self.lr = 1e-3 if lr is None else lr
        self.wd = 1e-5 if wd is None else wd
        self.lr_scheduler_freq = 1 if lr_scheduler_freq is None else lr_scheduler_freq
        self.lr_scheduler_interval = (
            "epoch" if lr_scheduler_interval is None else lr_scheduler_interval
        )
        self.lr_scheduler_patience = (
            5 if lr_scheduler_patience is None else lr_scheduler_patience
        )
        self.lr_scheduler_factor = (
            0.2 if lr_scheduler_factor is None else lr_scheduler_factor
        )
        self.lr_scheduler_mode = lr_scheduler_mode or "plateau"
        self.lr_scheduler_max_lr = lr_scheduler_max_lr or (self.lr * 10)  # Default 10x current lr
        self.lr_scheduler_total_steps = lr_scheduler_total_steps
        self.lr_monitor_key = "val_loss" if lr_monitor_key is None else lr_monitor_key

        self.use_infer_top_hvgs=use_infer_top_hvgs
        self.use_mask = use_mask  # Unified mask switch for training loss and evaluation

        if datamodule is not None:

            self.datamodule = datamodule

            self.use_mix_pert=datamodule.use_mix_pert

            if self.use_mix_pert:
                self.gene_key=datamodule.gene_key
                self.drug_key=datamodule.drug_key
                self.env_key=datamodule.env_key
                self.gene_pert_dim=datamodule.train_dataset.transform.gene_pert_dim
                self.drug_pert_dim=datamodule.train_dataset.transform.drug_pert_dim
                self.env_pert_dim=datamodule.train_dataset.transform.env_pert_dim
                self.crispr_pert_dim=datamodule.train_dataset.transform.crispr_pert_dim
                
                if datamodule.train_dataset.transform.use_covs:
                    self.cov_keys=datamodule.train_dataset.transform.cov_keys
                    self.cov_dims=datamodule.train_dataset.transform.cov_dims
                else:
                    self.cov_keys=[]
                    self.cov_dims={}
            else:
                self.pert_key = datamodule.pert_key
                self.cov_keys = datamodule.cov_keys
                self.cov_dims = {}

            self.result_avg_keys=datamodule.result_avg_keys
            self.control_val = datamodule.control_val

            self.gene_names=datamodule.train_dataset.get_gene_names()
            self.n_genes=len(self.gene_names)
            self.embedding_dim=datamodule.train_dataset.get_embedding_width()

            self.evaluation_config = datamodule.evaluation

            if self.use_infer_top_hvgs and hasattr(datamodule, "inference_top_hvg"):
                self.infer_gene_ids=datamodule.inference_top_hvg
            
            self.mask_type=self.datamodule.mask_type
            self.cellclass_mask_dict=self.datamodule.cellclass_mask_dict

    def _ensure_2d(self, t: torch.Tensor | None) -> torch.Tensor | None:
        """Convert [B,S,G] -> [B*S,G], keep [N,G] as-is."""
        if t is None:
            return None
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)
        if t.dim() == 2:
            return t
        if t.dim() == 3:
            return t.reshape(-1, t.size(-1))
        raise ValueError(f"Expected 2D or 3D tensor, got dim={t.dim()}, shape={tuple(t.shape)}")

    def _get_mask(self, batch) -> torch.Tensor:
        if not self.use_mask:
            return None
        return batch.mask
    
    def  auto_mse(self, pred, target,mask=None):
        import torch.nn.functional as F
        if mask is not None:
            masked_loss = F.mse_loss(pred*mask, target*mask, reduction="none")
            valid = mask.sum(dim=1)
            loss_per_batch = (masked_loss * mask).sum(dim=1)
            loss = (loss_per_batch / valid).nanmean()
        else:
            loss = F.mse_loss(pred, target, reduction="mean")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )

        if self.lr_scheduler_mode == "onecycle":
            # OneCycleLR: requires max_lr and total steps
            max_lr = self.lr_scheduler_max_lr
            if max_lr is None:
                max_lr = self.lr 

            total_steps = self.lr_scheduler_total_steps
            if total_steps is None and hasattr(self, 'trainer') and self.trainer is not None:
                # Dynamically calculate total steps: steps_per_epoch * max_epochs
                try:
                    if hasattr(self.trainer, 'max_epochs') and hasattr(self.trainer.datamodule, 'train_dataloader'):
                        # Get length of training dataloader
                        train_dl = self.trainer.datamodule.train_dataloader()
                        steps_per_epoch = len(train_dl)
                        total_steps = steps_per_epoch * self.trainer.max_epochs
                        print(f"OneCycleLR: dynamically calculated total_steps = {steps_per_epoch} * {self.trainer.max_epochs} = {total_steps}")
                except Exception as e:
                    print(f"Could not calculate total_steps dynamically: {e}")
                    total_steps = 100 * 100  # fallback
            elif total_steps is None:
                # Default assumption: 100 epochs, 100 steps per epoch
                total_steps = 100 * 100
                print(f"OneCycleLR: using default total_steps = {total_steps}")

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                anneal_strategy='cos',
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",  # OneCycleLR based on step
            }
        elif self.lr_scheduler_mode == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=getattr(self, 'lr_scheduler_step_size', None) or 10,  # Decrease every N epochs
                gamma=getattr(self, 'lr_scheduler_gamma', None) or 0.1,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        else:  # Default to ReduceLROnPlateau
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

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def on_test_start(self) -> None:
        super().on_test_start()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.preds_list=[]
        self.unique_aggregations=set()
        for eval_dict in self.evaluation_config.evaluation_pipelines:
            self.unique_aggregations.add(eval_dict["aggregation"])
        self.summary_metrics=None

    def test_step(
        self,
        data_tuple:tuple[any,pd.DataFrame],
        batch_idx: int,
    ):

        batch,obs_df=data_tuple
        predicted_expression = self.predict(batch)
        # Only convert to numpy for storage at the end (move to CPU only when needed)
        # This minimizes CPU-GPU transfers
        if isinstance(predicted_expression, torch.Tensor):
            # Detach to avoid gradient computation, move to CPU only for storage
            pred_np = predicted_expression.detach().cpu().numpy()
        else:
            pred_np = np.asarray(predicted_expression)
        self.preds_list.append((pred_np, obs_df))

    def predict(self, batch):
        pass

    def _gather_predictions(self):
        """Gather predictions from all distributed ranks."""
        import torch.distributed as dist

        local_expr = np.concatenate([expr for expr, _ in self.preds_list])
        local_obs = pd.concat([obs for _, obs in self.preds_list])

        is_distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_distributed else 1
        rank = dist.get_rank() if is_distributed else 0

        gathered_data = [None for _ in range(world_size)]

        if is_distributed:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            dist.all_gather_object(gathered_data, (local_expr, local_obs))
        else:
            gathered_data[0] = (local_expr, local_obs)

        return gathered_data, is_distributed, rank

    def _build_anndata(self, gathered_data):
        """Build predicted and reference AnnData objects from gathered data."""
        gathered_expr = np.concatenate([expr for expr, _ in gathered_data])
        gathered_obs = pd.concat([obs for _, obs in gathered_data], ignore_index=True)

        gene_names = self.gene_names
        if hasattr(self, 'infer_gene_ids'):
            gene_names = gene_names[self.infer_gene_ids]

        control_adata = self.datamodule.test_dataset.control_adata[:, gene_names]
        pert_adata = self.datamodule.test_dataset.pert_adata[:, gene_names]

        predicted_adata = ad.AnnData(
            X=gathered_expr,
            obs=gathered_obs,
            var=pd.DataFrame(index=gene_names),
        )
        predicted_adata = ad.concat([predicted_adata, control_adata])
        predicted_adata.obs_names_make_unique()
        reference_adata = ad.concat([pert_adata, control_adata])

        return predicted_adata, reference_adata, gene_names

    def _compute_sample_level_pcc(self, predicted_adata, reference_adata, eval_features):
        """
        Compute sample-level Pearson Correlation Coefficient (PCC) without aggregation.
        Calculate PCC between predictions and ground truth for each sample, then return the average.
        
        Args:
            predicted_adata: Predicted AnnData object
            reference_adata: Reference AnnData object
            eval_features: Subset of genes used for evaluation
            
        Returns:
            float: Average PCC across all samples
        """
        from scipy.stats import pearsonr
        
        # Get gene indices corresponding to eval_features
        gene_mask = np.isin(predicted_adata.var_names, eval_features)
        
        # Extract predicted and reference expression matrices
        pred_X = np.asarray(predicted_adata.X[:, gene_mask])
        ref_X = np.asarray(reference_adata.X[:, gene_mask])
        
        # Ensure same number of samples
        n_samples = min(pred_X.shape[0], ref_X.shape[0])
        
        # Calculate PCC for each sample
        pcc_values = []
        for i in range(n_samples):
            pred_row = pred_X[i].flatten()
            ref_row = ref_X[i].flatten()
            
            # Skip all-zero or constant rows
            if np.std(pred_row) > 0 and np.std(ref_row) > 0:
                pcc, _ = pearsonr(pred_row, ref_row)
                if not np.isnan(pcc):
                    pcc_values.append(pcc)
        
        # Return average PCC
        if len(pcc_values) > 0:
            return float(np.mean(pcc_values))
        else:
            return 0.0

    def _compute_ot_distances(self, predicted_adata, reference_adata, eval_features):
        """
        Compute distribution distance metrics:
        1) Energy Distance (hyperparameter-free statistical distance)
        2) Sinkhorn Divergence (GeomLoss, Wasserstein-2)

        Average after grouping by cov_pert
        """
        import numpy as np
        import torch
        from scipy.spatial.distance import cdist
        from geomloss import SamplesLoss

        # ================= Energy Distance =================
        def energy_distance(X, Y):
            d_xy = cdist(X, Y, metric='euclidean')
            d_xx = cdist(X, X, metric='euclidean')
            d_yy = cdist(Y, Y, metric='euclidean')

            n = len(X)
            m = len(Y)

            term_xy = 2.0 * d_xy.mean()
            term_xx = d_xx.sum() / (n * (n - 1))
            term_yy = d_yy.sum() / (m * (m - 1))

            return term_xy - term_xx - term_yy

        # ================= Sinkhorn (GeomLoss) =================
        device = "cuda" if torch.cuda.is_available() else "cpu"

        sinkhorn_loss = SamplesLoss(
            loss="sinkhorn",   # Sinkhorn divergence (debiased)
            p=2,               # squared Euclidean cost → Wasserstein-2
            blur=0.05,         # Regularization strength (default robust)
            scaling=0.9,
            backend="tensorized"
        )

        def sinkhorn_distance_geomloss(X, Y):
            X_t = torch.tensor(X, dtype=torch.float32, device=device)
            Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
            return sinkhorn_loss(X_t, Y_t).item()

        # ================= Data Preparation =================
        gene_mask = np.isin(predicted_adata.var_names, eval_features)

        pred_X = np.asarray(predicted_adata.X[:, gene_mask])
        ref_X = np.asarray(reference_adata.X[:, gene_mask])

        pert_col = '_merged_pert_col_' if self.use_mix_pert else self.pert_key
        cov_cols = [k for k in self.cov_keys if k not in self.result_avg_keys]

        def get_group_key(obs, idx):
            parts = [str(obs[pert_col].iloc[idx])]
            for c in cov_cols:
                parts.append(str(obs[c].iloc[idx]))
            return "_".join(parts)

        # Grouping
        pred_groups = {}
        for i in range(len(predicted_adata)):
            key = get_group_key(predicted_adata.obs, i)
            pred_groups.setdefault(key, []).append(i)

        ref_groups = {}
        for i in range(len(reference_adata)):
            key = get_group_key(reference_adata.obs, i)
            ref_groups.setdefault(key, []).append(i)

        # ================= Group-wise Calculation =================
        energy_values = []
        sinkhorn_values = []

        common_groups = set(pred_groups.keys()) & set(ref_groups.keys())

        for group_key in common_groups:
            pred_idx = pred_groups[group_key]
            ref_idx = ref_groups[group_key]

            pred_samples = pred_X[pred_idx]
            ref_samples = ref_X[ref_idx]

            if len(pred_samples) < 2 or len(ref_samples) < 2:
                continue

            try:
                e_dist = energy_distance(pred_samples, ref_samples)
                s_dist = sinkhorn_distance_geomloss(pred_samples, ref_samples)

                energy_values.append(float(e_dist))
                sinkhorn_values.append(float(s_dist))

            except Exception:
                continue

        mean_energy = float(np.mean(energy_values)) if energy_values else 0.0
        mean_sinkhorn = float(np.mean(sinkhorn_values)) if sinkhorn_values else 0.0

        return mean_energy, mean_sinkhorn

    def _compute_pca_metrics(self, predicted_adata, reference_adata, eval_features, model_name, n_components=50):
        """
        Compute metrics in PCA reduced space: Evaluation aggregation metrics, sample-level PCC, Energy Distance, Sinkhorn Divergence.
        
        Fit PCA on reference data, then project both predicted and reference data into this PCA space for evaluation.
        Reuse Evaluation, _compute_sample_level_pcc, and _compute_ot_distances functions for calculations.
        All metric names have "pca_" prefix.
        
        Args:
            predicted_adata: Predicted AnnData object
            reference_adata: Reference AnnData object
            eval_features: Subset of genes used for evaluation
            model_name: Model name for Evaluation
            n_components: Number of PCA dimensions (default: 50)
            
        Returns:
            dict: Dictionary containing the following metrics (all metric names have "pca_" prefix)
                - pca_{metric}_{aggr}: Aggregation metrics computed by Evaluation
                - pca_{metric}_rank_{aggr}: Ranking metrics computed by Evaluation (if rank is configured)
                - pca_pcc_no_aggr: Sample-level PCC in PCA space
                - pca_energy_distance: Energy Distance in PCA space
                - pca_sinkhorn_divergency: Sinkhorn Divergence in PCA space
        """
        from sklearn.decomposition import PCA
        
        # ================= Data Preparation =================
        gene_mask = np.isin(predicted_adata.var_names, eval_features)
        pred_X = np.asarray(predicted_adata.X[:, gene_mask])
        ref_X = np.asarray(reference_adata.X[:, gene_mask])
        
        # Determine actual PCA dimensions (not exceeding number of features and samples)
        actual_n_components = min(n_components, pred_X.shape[1], ref_X.shape[0] - 1)
        
        # ================= PCA Dimensionality Reduction =================
        # Fit PCA on reference data
        pca = PCA(n_components=actual_n_components)
        ref_pca = pca.fit_transform(ref_X)
        pred_pca = pca.transform(pred_X)
        
        # ================= Build Temporary AnnData for PCA Space =================
        # Create PCA dimension feature names (passed as eval_features to reuse functions)
        pca_feature_names = [f"PC{i+1}" for i in range(actual_n_components)]
        
        # Build temporary AnnData containing PCA data, preserving original obs information
        pred_pca_adata = ad.AnnData(
            X=pred_pca,
            obs=predicted_adata.obs.copy(),
            var=pd.DataFrame(index=pca_feature_names),
        )
        ref_pca_adata = ad.AnnData(
            X=ref_pca,
            obs=reference_adata.obs.copy(),
            var=pd.DataFrame(index=pca_feature_names),
        )
        
        pca_metrics_dict = {}
        
        # ================= 1. Compute Aggregation Metrics using Evaluation (PCA Space) =================
        cov_cols = [k for k in self.cov_keys if k not in self.result_avg_keys]
        
        ev = Evaluation(
            model_adatas=[pred_pca_adata],
            model_names=[model_name],
            ref_adata=ref_pca_adata,
            pert_col='_merged_pert_col_' if self.use_mix_pert else self.pert_key,
            cov_cols=cov_cols,
            ctrl=self.control_val,
            features=pca_feature_names,
        )
        
        for aggr in self.unique_aggregations:
            ev.aggregate(aggr_method=aggr)
        
        for eval_dict in self.evaluation_config.evaluation_pipelines:
            aggr = eval_dict["aggregation"]
            metric = eval_dict["metric"]
            ev.evaluate(aggr_method=aggr, metric=metric)
            
            df = ev.evals[aggr][metric].copy()
            avg = df.groupby("model").mean("metric")
            pca_metrics_dict[f"pca_{metric}_{aggr}"] = avg["metric"].iloc[0]
            
            if eval_dict.get("rank"):
                ev.evaluate_pairwise(aggr_method=aggr, metric=metric)
                ev.evaluate_rank(aggr_method=aggr, metric=metric)
                rank_df = ev.rank_evals[aggr][metric].copy()
                avg_rank = rank_df.groupby("model").mean("rank")
                pca_metrics_dict[f"pca_{metric}_rank_{aggr}"] = avg_rank["rank"].iloc[0]
        
        # ================= 2. Sample-level PCC (PCA Space) =================
        pca_pcc = self._compute_sample_level_pcc(pred_pca_adata, ref_pca_adata, pca_feature_names)
        pca_metrics_dict["pca_pcc_no_aggr"] = pca_pcc
        
        # ================= 3. OT Distances (PCA Space) =================
        pca_energy, pca_sinkhorn = self._compute_ot_distances(pred_pca_adata, ref_pca_adata, pca_feature_names)
        pca_metrics_dict["pca_energy_distance"] = pca_energy
        pca_metrics_dict["pca_sinkhorn_divergency"] = pca_sinkhorn
        
        return pca_metrics_dict

    def _compute_deg_metrics(self, predicted_adata, reference_adata, eval_features, top_n_deg=50):
        """
        Compute DEG (Differentially Expressed Genes) related metrics: IoU, Precision, Recall.
        
        For each perturbation:
        1. Compute top DEG for the perturbation relative to control in predicted_adata (sorted by absolute expression difference)
        2. Compute top DEG for the perturbation relative to control in reference_adata
        3. Compute IoU, Precision, Recall for the two DEG sets
        4. Average across all perturbations
        
        Args:
            predicted_adata: Predicted AnnData object
            reference_adata: Reference AnnData object
            eval_features: Subset of genes used for evaluation
            top_n_deg: Number of top DEG to select per perturbation (default: 50)
            
        Returns:
            tuple: (mean_iou, mean_precision, mean_recall)
                - IoU = |pred_DEG ∩ ref_DEG| / |pred_DEG ∪ ref_DEG|
                - Precision = |pred_DEG ∩ ref_DEG| / |pred_DEG|  (How many predicted DEG are in ground truth DEG)
                - Recall = |pred_DEG ∩ ref_DEG| / |ref_DEG|  (How many ground truth DEG are captured by predicted DEG)
        """
        # Get pert column name
        pert_col = '_merged_pert_col_' if self.use_mix_pert else self.pert_key
        
        # Get gene indices corresponding to eval_features
        gene_mask = np.isin(predicted_adata.var_names, eval_features)
        
        # Extract expression matrices
        pred_X = np.asarray(predicted_adata.X[:, gene_mask])
        ref_X = np.asarray(reference_adata.X[:, gene_mask])
        
        # Get gene names
        genes = np.array(predicted_adata.var_names)[gene_mask]
        
        # Separate control and perturbation samples
        pred_ctrl_mask = (predicted_adata.obs[pert_col] == self.control_val).values
        ref_ctrl_mask = (reference_adata.obs[pert_col] == self.control_val).values
        
        # Calculate average control expression
        pred_ctrl_mean = pred_X[pred_ctrl_mask].mean(axis=0) if pred_ctrl_mask.sum() > 0 else np.zeros(pred_X.shape[1])
        ref_ctrl_mean = ref_X[ref_ctrl_mask].mean(axis=0) if ref_ctrl_mask.sum() > 0 else np.zeros(ref_X.shape[1])
        
        # Get all perturbations (exclude control)
        all_perts = set(predicted_adata.obs[pert_col].unique()) | set(reference_adata.obs[pert_col].unique())
        perts = [p for p in all_perts if p != self.control_val]
        
        iou_values = []
        precision_values = []
        recall_values = []
        
        for pert in perts:
            # Get sample mask for this pert
            pred_pert_mask = (predicted_adata.obs[pert_col] == pert).values
            ref_pert_mask = (reference_adata.obs[pert_col] == pert).values
            
            # Skip non-existent pert
            if pred_pert_mask.sum() == 0 or ref_pert_mask.sum() == 0:
                continue
            
            # Calculate average expression for this pert
            pred_pert_mean = pred_X[pred_pert_mask].mean(axis=0)
            ref_pert_mean = ref_X[ref_pert_mask].mean(axis=0)
            
            # Calculate differences (absolute difference relative to control)
            pred_diff = np.abs(pred_pert_mean - pred_ctrl_mean)
            ref_diff = np.abs(ref_pert_mean - ref_ctrl_mean)
            
            # Determine actual top_n (not exceeding total number of genes)
            actual_top_n = min(top_n_deg, len(genes))
            
            # Select top N DEG (sorted by absolute difference, take largest N)
            pred_top_indices = np.argsort(pred_diff)[-actual_top_n:]
            ref_top_indices = np.argsort(ref_diff)[-actual_top_n:]
            
            pred_deg_set = set(genes[pred_top_indices])
            ref_deg_set = set(genes[ref_top_indices])
            
            # Calculate IoU, Precision, Recall
            intersection = len(pred_deg_set & ref_deg_set)
            union = len(pred_deg_set | ref_deg_set)
            
            iou = intersection / union if union > 0 else 0.0
            # Precision: How many predicted DEG are in ground truth DEG
            precision = intersection / len(pred_deg_set) if len(pred_deg_set) > 0 else 0.0
            # Recall: How many ground truth DEG are captured by predicted DEG
            recall = intersection / len(ref_deg_set) if len(ref_deg_set) > 0 else 0.0
            
            iou_values.append(iou)
            precision_values.append(precision)
            recall_values.append(recall)
        
        mean_iou = float(np.mean(iou_values)) if iou_values else 0.0
        mean_precision = float(np.mean(precision_values)) if precision_values else 0.0
        mean_recall = float(np.mean(recall_values)) if recall_values else 0.0
        
        return mean_iou, mean_precision, mean_recall

    def _run_evaluation(self, predicted_adata, reference_adata, eval_features, model_name):
        """Run evaluation and return summary metrics."""
        if eval_features is None:
            eval_features = reference_adata.var_names
            
        cov_cols = [k for k in self.cov_keys if k not in self.result_avg_keys]
        
        ev = Evaluation(
            model_adatas=[predicted_adata],
            model_names=[model_name],
            ref_adata=reference_adata,
            pert_col='_merged_pert_col_' if self.use_mix_pert else self.pert_key,
            cov_cols=cov_cols,
            ctrl=self.control_val,
            features=eval_features,
        )

        for aggr in self.unique_aggregations:
            ev.aggregate(aggr_method=aggr)

        summary_metrics_dict = {}
        for eval_dict in self.evaluation_config.evaluation_pipelines:
            aggr = eval_dict["aggregation"]
            metric = eval_dict["metric"]
            ev.evaluate(aggr_method=aggr, metric=metric)

            df = ev.evals[aggr][metric].copy()
            avg = df.groupby("model").mean("metric")
            summary_metrics_dict[f"{metric}_{aggr}"] = avg["metric"]

            if eval_dict.get("rank"):
                ev.evaluate_pairwise(aggr_method=aggr, metric=metric)
                ev.evaluate_rank(aggr_method=aggr, metric=metric)
                rank_df = ev.rank_evals[aggr][metric].copy()
                avg_rank = rank_df.groupby("model").mean("rank")
                summary_metrics_dict[f"{metric}_rank_{aggr}"] = avg_rank["rank"]

        # ====== Sample-level PCC (no-aggr) calculation, without using evaluation package ======
        sample_pcc = self._compute_sample_level_pcc(predicted_adata, reference_adata, eval_features)
        summary_metrics_dict["pcc_no_aggr"] = pd.Series({model_name: sample_pcc})

        # ====== OT distance calculation (MMD, Sinkhorn), averaged by cov_pert group ======
        try:
            mean_mmd, mean_sinkhorn = self._compute_ot_distances(predicted_adata, reference_adata, eval_features)
            summary_metrics_dict["energy_distance"] = pd.Series({model_name: mean_mmd})
            summary_metrics_dict["sinkhorn_divergency"] = pd.Series({model_name: mean_sinkhorn})
        except Exception as e:
            print(f"Warning: OT distance computation failed: {e}")
            summary_metrics_dict["energy_distance"] = pd.Series({model_name: 0.0})
            summary_metrics_dict["sinkhorn_divergency"] = pd.Series({model_name: 0.0})

        # ====== Metrics calculation in PCA space (Evaluation aggregation metrics, PCC, Energy Distance, Sinkhorn) ======
        try:
            pca_metrics = self._compute_pca_metrics(
                predicted_adata, reference_adata, eval_features, model_name, n_components=50
            )
            # Add all PCA metrics to summary_metrics_dict (already with "pca_" prefix)
            for metric_name, metric_value in pca_metrics.items():
                summary_metrics_dict[metric_name] = pd.Series({model_name: metric_value})
        except Exception as e:
            print(f"Warning: PCA metrics computation failed: {e}")
            # Set default PCA metrics to 0.0
            summary_metrics_dict["pca_pcc_no_aggr"] = pd.Series({model_name: 0.0})
            summary_metrics_dict["pca_energy_distance"] = pd.Series({model_name: 0.0})
            summary_metrics_dict["pca_sinkhorn_divergency"] = pd.Series({model_name: 0.0})

        # ====== DEG metrics calculation (IoU, Precision, Recall), averaged by pert group ======
        try:
            deg_iou, deg_precision, deg_recall = self._compute_deg_metrics(
                predicted_adata, reference_adata, eval_features, top_n_deg=50
            )
            summary_metrics_dict["deg_iou"] = pd.Series({model_name: deg_iou})
            summary_metrics_dict["deg_precision"] = pd.Series({model_name: deg_precision})
            summary_metrics_dict["deg_recall"] = pd.Series({model_name: deg_recall})
        except Exception as e:
            print(f"Warning: DEG metrics computation failed: {e}")
            summary_metrics_dict["deg_iou"] = pd.Series({model_name: 0.0})
            summary_metrics_dict["deg_precision"] = pd.Series({model_name: 0.0})
            summary_metrics_dict["deg_recall"] = pd.Series({model_name: 0.0})

        summary_metrics = pd.DataFrame(summary_metrics_dict).T.applymap(
            lambda x: float(np.format_float_positional(
                x, precision=4, unique=False, fractional=False, trim="k"
            ))
        )

        return ev, summary_metrics, summary_metrics_dict

    def _get_output_dir(self):
        """Get output directory from hydra or logger."""
        try:
            return HydraConfig.get().runtime.output_dir
        except Exception:
            if self.logger is not None:
                logger_obj = self.logger[0] if isinstance(self.logger, (list, tuple)) and len(self.logger) > 0 else self.logger
                return getattr(logger_obj, "save_dir", None) or self.evaluation_config.save_dir
            return self.evaluation_config.save_dir

    def _save_results(self, ev, summary_metrics, predicted_adata, output_dir):
        """Save evaluation results, summary metrics, and predictions."""
        summary_dir = os.path.join(output_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        os.makedirs(self.evaluation_config.save_dir, exist_ok=True)

        ev.save(self.evaluation_config.save_dir)

        # Save summary CSV
        csv_path = os.path.join(self.evaluation_config.save_dir, "summary.csv")
        summary_metrics.to_csv(csv_path, index_label="metric")

        ckpt_type = getattr(self, "current_test_ckpt_type", None)
        suffix = f"_{ckpt_type}" if ckpt_type and ckpt_type != "unknown" else ""
        
        summary_csv_path = os.path.join(summary_dir, f"summary_metrics{suffix}.csv")
        summary_metrics.to_csv(summary_csv_path, index_label="metric")

        # Save predictions
        pred_h5ad_path = os.path.join(summary_dir, f"predictions{suffix}.h5ad")
        try:
            predicted_adata.write(pred_h5ad_path)
        except OSError as e:
            if e.errno == 122:
                print(f"WARNING: Disk quota exceeded, skipping prediction save.")
            else:
                print(f"WARNING: Failed to save predictions: {e}")
        except Exception as e:
            print(f"WARNING: Failed to save predictions: {e}")

        return csv_path, summary_dir

    def _log_to_wandb(self, summary_metrics_dict):
        """Log metrics to wandb if enabled."""
        save_preds_to_wandb = self.evaluation_config.get("save_predictions_to_wandb", False)
        if not save_preds_to_wandb or self.logger is None:
            return

        try:
            loggers = self.logger if isinstance(self.logger, (list, tuple)) else [self.logger]
            for logger in loggers:
                if hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
                    test_metrics_dict = {f"test_{k}": float(v) for k, v in summary_metrics_dict.items()}
                    logger.experiment.log(test_metrics_dict)
        except Exception:
            pass  # Silently skip wandb logging errors

    def _run_evaluation_per_cellclass(self, predicted_adata, reference_adata, model_name, output_dir):
        """
        Run evaluation separately for each cellclass (using obs['cellclass'] and cellclass_mask_dict).
        
        This function determines different evaluation strategies based on the presence of cellclass grouping information in the data:
        - Without cellclass information: Perform unified evaluation on all data
        - With cellclass information: Evaluate each cellclass group separately and aggregate results
        
        Args:
            predicted_adata (AnnData): Model-predicted expression matrix, obs should contain 'cellclass' column (if applicable)
            reference_adata (AnnData): Reference (ground truth) expression matrix, obs should contain 'cellclass' column (if applicable)
            model_name (str): Model name used to identify the model in evaluation reports
            output_dir (str): Root path of output directory where evaluation results will be saved
        
        Returns:
            tuple: (summary_metrics, summary_metrics_dict, csv_path)
                - summary_metrics (pd.DataFrame | None): DataFrame of average evaluation metrics across all cellclasses
                - summary_metrics_dict (dict): Metric dictionary grouped by cellclass {cellclass: {metric: value}}
                - csv_path (str | None): Save path of the aggregated CSV file
        
        Output directory structure (when cellclass grouping is present):
        -----------------------------------------------
        {output_dir}/
        ├── cellclass_evaluation/           # Cellclass-specific evaluation results
        │   ├── {cellclass_1}/              # Results directory for the first cellclass
        │   │   ├── summary.csv             # Evaluation metric summary for this cellclass
        │   │   ├── predictions.h5ad        # Predicted results AnnData for this cellclass
        │   │   └── ... (Other files generated by Evaluation.save())
        │   ├── {cellclass_2}/              # Results directory for the second cellclass
        │   │   └── ...
        │   └── ...
        └── summary/                        # Aggregation directory
            ├── summary_by_cellclass.csv    # Detailed metrics for all cellclasses (with cellclass column)
            └── summary_avg.csv             # Average metrics across all cellclasses
        
        Output directory structure (when no cellclass grouping):
        -----------------------------------------------
        {output_dir}/
        └── summary/
            ├── summary_metrics_{ckpt_type}.csv  # Evaluation metric summary
            └── predictions_{ckpt_type}.h5ad     # Predicted results AnnData
        {evaluation_config.save_dir}/
            ├── summary.csv                      # Evaluation metric summary (copy)
            └── ... (Other files generated by Evaluation.save())
        """
        # ========== Step 1: Check if cellclass-based evaluation is needed ==========
        # If there's no 'cellclass' column in reference_adata.obs or no cellclass_mask_dict configured,
        # use full evaluation mode and evaluate all data uniformly
        if 'cellclass' not in reference_adata.obs or not getattr(self, 'cellclass_mask_dict', None):
            ev, metrics, mdict = self._run_evaluation(
                predicted_adata, reference_adata, None, model_name
            )
            return metrics, mdict, self._save_results(ev, metrics, predicted_adata, output_dir)[0]
        
        # ========== Step 2: Get list of cellclasses common to both predicted and reference data ==========
        # Take intersection to ensure only cellclasses present in both are evaluated, sorted alphabetically
        groups = sorted(set(predicted_adata.obs['cellclass'].unique()) & set(reference_adata.obs['cellclass'].unique()))
        all_metrics, all_mdict = {}, {}  # Store evaluation results for each cellclass
        cc_dir = os.path.join(output_dir, "cellclass_evaluation")  # Root directory for cellclass evaluation results
        
        # ========== Step 3: Evaluate each cellclass independently ==========
        for cc in groups:
            # 3.1 Filter subset by cellclass
            _cellclass_mask=self.cellclass_mask_dict.get(cc,None)
            
            pred_sub = predicted_adata[predicted_adata.obs['cellclass'] == cc]
            ref_sub = reference_adata[reference_adata.obs['cellclass'] == cc]
            
            # 3.2 Skip empty datasets (no predicted or reference samples)
            if pred_sub.n_obs == 0 or ref_sub.n_obs == 0:
                continue
            
            try:
                # 3.4 Run evaluation on this cellclass subset
                # Ensure _cellclass_mask is a 1D numpy array for correct gene_names indexing
                if _cellclass_mask is not None:
                    _cellclass_mask = np.asarray(_cellclass_mask).flatten()
                gene_names_arr = np.array(self.gene_names)
                eval_gene_names = gene_names_arr[_cellclass_mask] if _cellclass_mask is not None else gene_names_arr
                ev, metrics, mdict = self._run_evaluation(pred_sub, ref_sub,
                                                          eval_gene_names, model_name)
                # 3.5 Create save directory for this cellclass (replace '/' with '_' to avoid path issues)
                save_dir = os.path.join(cc_dir, str(cc).replace('/', '_'))
                os.makedirs(save_dir, exist_ok=True)
                
                # 3.6 Save evaluation results
                ev.save(save_dir)  # Save full Evaluation object results
                metrics.to_csv(os.path.join(save_dir, "summary.csv"), index_label="metric")  # Save metric summary
                
                # 3.7 Try to save predicted AnnData (may fail due to disk space etc., handle silently)
                try:
                    pred_sub.write(os.path.join(save_dir, "predictions.h5ad"))
                except Exception:
                    pass
                
                # 3.8 Record evaluation results for this cellclass
                all_metrics[cc], all_mdict[cc] = metrics, mdict
                print(f"[{cc}] genes={pred_sub.n_vars}, saved to {save_dir}")
                
            except Exception as e:
                # Record failed cellclass (don't interrupt overall process)
                print(f"[{cc}] failed: {e}")
        
        # ========== Step4：Check whether successful evaluation results exist ==========
        if not all_metrics:
            return None, {}, None
        
        # ========== Step5：Aggregate evaluation results for all cell classes ==========
        summary_dir = os.path.join(output_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        # 5.1 Save detailed metrics grouped by cell class (each row includes a cell class label)
        pd.concat([df.assign(cellclass=cc) for cc, df in all_metrics.items()]).to_csv(
            os.path.join(summary_dir, "summary_by_cellclass.csv"), index_label="metric"
        )
        
        # 5.2 Compute and save average metrics across all cell classes
        avg = pd.concat(all_metrics.values()).groupby(level=0).mean()
        avg.to_csv(os.path.join(summary_dir, "summary_avg.csv"), index_label="metric")
        
        return avg, all_mdict, os.path.join(summary_dir, "summary_by_cellclass.csv")

    def on_test_end(self) -> None:
        """
        Callback invoked at the end of testing to aggregate predictions and run evaluation.
        
        This function is a PyTorch Lightning lifecycle hook, automatically called after all test batches are processed.
        Main responsibilities:
        1. Collect prediction results from all distributed processes
        2. Build AnnData objects and run evaluation
        3. Save evaluation results and prediction data
        4. Synchronize evaluation metrics to all processes
        5. Release memory resources
        
        Distributed training notes:
        ----------------
        - In multi-GPU/multi-node training, each process only holds part of the predictions
        - This function uses all_gather_object to collect predictions from all processes
        - Evaluation runs only on rank 0, then results are broadcast to other processes
        - Barrier synchronization ensures all processes are aligned before continuing
        
        Complete output directory structure:
        ----------------
        {output_dir}/                           # Root directory determined by Hydra or Logger
        ├── cellclass_evaluation/               # Evaluation results grouped by cell class (if applicable)
        │   ├── {cellclass_1}/
        │   │   ├── summary.csv                 # Evaluation metrics for this cell class
        │   │   ├── predictions.h5ad            # Predicted AnnData for this cell class
        │   │   ├── aggregations/               # Evaluation.save() Aggregation results generated by Evaluation.save()
        │   │   │   ├── {aggr_method_1}.h5ad
        │   │   │   └── ...
        │   │   └── evaluations/                # Evaluation.save() Evaluation details generated by Evaluation.save()
        │   │       ├── {aggr_method}_{metric}.csv
        │   │       └── ...
        │   ├── {cellclass_2}/
        │   │   └── ...
        │   └── ...
        └── summary/                            # Summary directory
            ├── summary_by_cellclass.csv        # Detailed metrics for all cell classes (when grouped by cell class)
            ├── summary_avg.csv                 # Average metrics across all cell classes (when grouped by cell class)
            ├── summary_metrics_{ckpt_type}.csv # Evaluation metric summary (without cell-class grouping)
            └── predictions_{ckpt_type}.h5ad    # Full prediction results (without cell-class grouping)
        
        {evaluation_config.save_dir}/           # Evaluation output directory specified by config
        ├── summary.csv                         # Copy of evaluation metric summary
        ├── aggregations/                       # Aggregated expression data
        │   └── ...
        └── evaluations/                        # Detailed evaluation results
            └── ...
        
        Attributes Modified:
            self.summary_metrics: Updated to the final evaluation metrics DataFrame
            self.preds_list: Cleared to release memory
        
        Note:
            - This function depends on self.preds_list initialized in on_test_start()
            - It depends on predictions appended by test_step() after each batch
        """
        import torch.distributed as dist
        super().on_test_end()
        
        # ========== Step1：Preparation ==========
        # Extract model name from class name (used as evaluation report identifier)
        # Example: <class 'VCBench.models.MyModel'> -> "MyModel"
        model_name = str(self.__class__).split(".")[-1].replace("'>", "")
        
        # Collect prediction results from all distributed processes
        # gathered_data: List[(np.ndarray, pd.DataFrame)]，each element is one process's (expression matrix, obs DataFrame)
        # is_distributed: bool，whether running in distributed mode
        # rank: int，current process rank (0 is the main process)
        gathered_data, is_distributed, rank = self._gather_predictions()
        summary_metrics, summary_metrics_dict = None, {}

        # ========== Step2：Run evaluation on the main process (rank 0) ==========
        # Run evaluation only on rank 0 to avoid duplicate computation and I/O conflicts
        if rank == 0:
            # 2.1 Build AnnData objects
            # predicted_adata: Model-predicted expression matrix + control-group data
            # reference_adata: Ground-truth perturbed data + control-group data
            predicted_adata, reference_adata, _ = self._build_anndata(gathered_data)
            
            # 2.2 Resolve output directory (prefer Hydra config, otherwise Logger or config file)
            output_dir = self._get_output_dir()

            # 2.3 Run evaluation by cell class (or globally, depending on data config)
            summary_metrics, summary_metrics_dict, csv_path = self._run_evaluation_per_cellclass(
                predicted_adata, reference_adata, model_name, output_dir
            )

            # 2.4 Print evaluation summary (if enabled by config)
            if self.evaluation_config.print_summary and summary_metrics is not None:
                print(f"\n===== Average Summary Metrics =====\n{summary_metrics}\n")
            if csv_path:
                print(f"Evaluation finished. Results saved to {csv_path}")

            # 2.5 Log to Weights & Biases (if enabled)
            # Log per-cell-class metrics first, then log averages across all cell classes
            if summary_metrics_dict:
                # 2.5.1 Log per-cell-class metrics (prefixed with cell class)
                for cc, cc_metrics in summary_metrics_dict.items():
                    cc_prefix = str(cc).replace('/', '_')
                    self._log_to_wandb({f"{cc_prefix}/{k}": v for k, v in cc_metrics.items()})

                # 2.5.2 Compute and log average metrics across all cell classes
                avg_dict = {}
                for cc_metrics in summary_metrics_dict.values():
                    for k, v in cc_metrics.items():
                        avg_dict.setdefault(k, []).append(float(v) if hasattr(v, '__float__') else v)
                self._log_to_wandb({f"mean/{k}": np.mean(v) for k, v in avg_dict.items()})

        # ========== Step3：Synchronize evaluation results to all processes ==========
        # In distributed settings, ensure all processes can access summary_metrics
        # This is important for downstream logic that may depend on evaluation results (e.g., model selection, early stopping)
        if is_distributed:
            # Use broadcast_object_list to broadcast summary_metrics from rank 0 to all processes
            obj_list = [summary_metrics]
            dist.broadcast_object_list(obj_list, src=0)
            self.summary_metrics = obj_list[0]
            
            # Synchronization barrier: ensure all processes receive the broadcast before continuing
            # Prevent fast processes from entering the next stage too early (e.g., starting a new training epoch)
            dist.barrier()
        else:
            # In non-distributed mode, direct assignment is sufficient
            self.summary_metrics = summary_metrics

        # ========== Step4：Clean up resources ==========
        # Release memory used by prediction lists (can be large, especially with large test sets)
        self.preds_list = []
        
        # Manually trigger garbage collection to release memory promptly
        # Especially important in GPU training to avoid later failures due to insufficient memory
        gc.collect()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):

        # Case 1: dict batch
        if isinstance(batch, dict):
            batch_dict=apply_to_collection(
                batch,
                torch.Tensor,
                lambda x: x.to(device)
            )
            return Batch(batch_dict)

        # Case 2: (dict, pandas_df)
        if isinstance(batch, tuple):
            batch_dict, obs_df = batch

            batch_dict = apply_to_collection(
                batch_dict,
                torch.Tensor,
                lambda x: x.to(device)
            )

            # Note: do not recursively apply to(device) to obs_df
            return Batch(batch_dict), obs_df

        return batch