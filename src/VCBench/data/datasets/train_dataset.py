import warnings
from pathlib import Path
from collections.abc import Callable, Mapping
import gc
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from omegaconf import DictConfig
from torch.utils.data import Dataset
import anndata as ad
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import random  # Python random is faster than np.random
from scipy.sparse import issparse


class scTrainDataset(Dataset):
    """
    High-performance version:
    - No AnnData slicing
    - All information pre-extracted to numpy arrays
    - No Pandas filtering
    """

    def __init__(
            self,
            pert_adata: ad.AnnData,
            control_adata: ad.AnnData,
            cov_keys: List[str],
            transform: DictConfig,
            pert_key: str|None=None,
            gene_key: str|None=None,
            drug_key:str|None=None,
            crispr_key: str|None=None,
            use_mix_pert: bool=False,
            env_key: str|None=None,
            embedding_key: str | None = None,
            raw_counts_key: str | None = None,
            cov_avg_sampling: bool = False,
            cell_set_len: int | None = None,
            add_keys: List[str] | None = None,
            predict_controls: bool = False,
            cellclass_mask_dict: Dict[str, np.ndarray] | None = None,
            mask_type: str|None=None,
            **kwargs
    ):
        super().__init__()

        self.use_mix_pert=use_mix_pert

        if predict_controls:
            pert_adata=ad.concat([pert_adata,control_adata])

        self.pert_adata=pert_adata
        self.control_adata=control_adata

        self.pert_obs=pert_adata.obs
        self.control_obs=control_adata.obs

        # -----------------------------------
        # 1. Extract all data in advance (greatly speed up)
        # -----------------------------------
        # If it's a sparse matrix, convert to dense array (faster random access)
        X_pert = pert_adata.X
        X_ctrl = control_adata.X
        self.X_pert = X_pert.toarray() if issparse(X_pert) else np.asarray(X_pert)
        self.X_ctrl = X_ctrl.toarray() if issparse(X_ctrl) else np.asarray(X_ctrl)
        
        # Convert to float32 and ensure contiguous memory layout (speed up tensor conversion)
        if self.X_pert.dtype != np.float32:
            self.X_pert = self.X_pert.astype(np.float32)
        if self.X_ctrl.dtype != np.float32:
            self.X_ctrl = self.X_ctrl.astype(np.float32)
        self.X_pert = np.ascontiguousarray(self.X_pert)
        self.X_ctrl = np.ascontiguousarray(self.X_ctrl)

        self.embedding_key = embedding_key
        if embedding_key:
            # Convert to float32 contiguous array (speed up tensor conversion)
            emb_pert = pert_adata.obsm[embedding_key]
            emb_ctrl = control_adata.obsm[embedding_key]
            self.emb_pert = np.ascontiguousarray(emb_pert, dtype=np.float32)
            self.emb_ctrl = np.ascontiguousarray(emb_ctrl, dtype=np.float32)
        else:
            self.emb_pert = None
            self.emb_ctrl = None

        self.raw_counts_key = raw_counts_key
        if raw_counts_key:
            self.raw_pert = pert_adata.obs[raw_counts_key].to_numpy()
            self.raw_ctrl = control_adata.obs[raw_counts_key].to_numpy()
        else:
            self.raw_pert = None
            self.raw_ctrl = None

        # -----------------------------------
        # 2. Build obs information in advance
        # -----------------------------------
        self.pert_key = pert_key
        self.gene_key = gene_key
        self.drug_key=drug_key
        self.env_key = env_key
        self.crispr_key = crispr_key
        self.cov_keys = cov_keys
        self.cov_avg_sampling = cov_avg_sampling
        self.cell_set_len = cell_set_len
        self.add_keys = add_keys
        self.mask_type = mask_type
        self.mask_dict = cellclass_mask_dict

        pert_obs = pert_adata.obs
        ctrl_obs = control_adata.obs

        # Combine cov information
        if cov_keys:
            pert_cov = self.merge_cols(pert_obs, cov_keys)
            ctrl_cov = self.merge_cols(ctrl_obs, cov_keys)
        else:
            pert_cov = np.array(["_"] * len(pert_obs))
            ctrl_cov = np.array(["_"] * len(ctrl_obs))

        self.pert_cov = pert_cov
        self.ctrl_cov = ctrl_cov

        # pert only
        if not self.use_mix_pert:
            self.pert_labels=pert_obs[pert_key].astype(str).to_numpy()
        else:
            # Process gene_key, convert to string type first to avoid Categorical issues
            gene_col = pert_obs[gene_key]
            if pd.api.types.is_categorical_dtype(gene_col):
                gene_col = gene_col.astype(str)
            self.gene_pert_labels = gene_col.fillna('').astype(str).to_numpy()

            # Process drug_key
            drug_col = pert_obs[drug_key]
            if pd.api.types.is_categorical_dtype(drug_col):
                drug_col = drug_col.astype(str)
            self.drug_pert_labels = drug_col.fillna('').astype(str).to_numpy()

            # Process env_key
            env_col = pert_obs[env_key]
            if pd.api.types.is_categorical_dtype(env_col):
                env_col = env_col.astype(str)
            self.env_pert_labels = env_col.fillna('').astype(str).to_numpy()

            # Process crispr_key
            crispr_col = pert_obs[crispr_key]
            if pd.api.types.is_categorical_dtype(crispr_col):
                crispr_col = crispr_col.astype(str)
            # CRISPR column may have many null values (only columns with gene_pt have CRISPR), convert NaN to empty string
            self.crispr_labels = crispr_col.fillna('').astype(str).to_numpy()

        # -----------------------------------
        # 3. Build indices (replace Pandas filtering)
        # -----------------------------------
        # (Important) Pre-build index lists for each cov / (cov,pert) group
        index_by_cov_pert_tmp = {}
        index_by_cov_ctrl_tmp = {}

        for i, c in enumerate(pert_cov):
            if self.use_mix_pert:
                key=(c,self.gene_pert_labels[i],self.drug_pert_labels[i],self.env_pert_labels[i],self.crispr_labels[i])
            else:
                key = (c, self.pert_labels[i])
            index_by_cov_pert_tmp.setdefault(key, []).append(i)

        for i, c in enumerate(ctrl_cov):
            index_by_cov_ctrl_tmp.setdefault(c, []).append(i)

        # weak check
        if any(len(v) == 0 for v in index_by_cov_ctrl_tmp.values()):
            print("⚠ warning: Some control cov are empty")

        # Convert to numpy array (speed up random.choice)
        self.index_by_cov_pert = {k: np.array(v, dtype=np.int64) for k, v in index_by_cov_pert_tmp.items()}
        self.index_by_cov_ctrl = {k: np.array(v, dtype=np.int64) for k, v in index_by_cov_ctrl_tmp.items()}

        # cov unique
        self.unique_covs = np.unique(pert_cov)
        self.unique_keys = list(self.index_by_cov_pert.keys())
        self._n_unique_keys = len(self.unique_keys)  # Cache length
        
        # Pre-split cov strings (avoid repeated split in build_output)
        self._cov_split_cache = {}
        for key in self.unique_keys:
            cov = key[0]
            if cov not in self._cov_split_cache:
                self._cov_split_cache[cov] = cov.split('<>')

        # -----------------------------------
        # 4. Transform initialization
        # -----------------------------------
        self.transform = transform(
            obs_df=pert_obs.copy()
        )

        # Record total length
        self.length = len(pert_obs) if predict_controls \
            else len(pert_obs)+ len(ctrl_obs)
            
        if self.mask_type=='cell':
            self.pert_expression_mask = np.ascontiguousarray((self.X_pert > 0).astype(np.float32))
        elif self.mask_type is not None and self.mask_dict is not None:
            mask = [self.mask_dict[cellclass] for cellclass in self.pert_adata.obs['cellclass']]
            self.pert_expression_mask = np.ascontiguousarray(np.stack(mask, axis=0), dtype=np.float32)
        else:
            self.pert_expression_mask = None

    def merge_cols(self, obs_df, cols):
        merged = obs_df[cols[0]].astype(str).to_numpy()
        for c in cols[1:]:
            merged = merged + "<>" + obs_df[c].astype(str).to_numpy()
        return merged

    def __len__(self):
        if self.cell_set_len:
            return self.length // self.cell_set_len
        else:
            return self.length

    def __getitem__(self, idx):
        if self.cell_set_len:
            return self.get_set_sample()
        else:
            return self.get_single_sample()

    # ---------------------------------------------------------
    # Sample a single cell (optimized version)
    # ---------------------------------------------------------
    def get_single_sample(self):
        # Use Python random (3-5x faster than np.random)
        key = self.unique_keys[random.randint(0, self._n_unique_keys - 1)]
        
        # Random selection from numpy array
        pert_arr = self.index_by_cov_pert[key]
        ctrl_arr = self.index_by_cov_ctrl[key[0]]
        pert_idx = pert_arr[random.randint(0, len(pert_arr) - 1)]
        ctrl_idx = ctrl_arr[random.randint(0, len(ctrl_arr) - 1)]

        if self.use_mix_pert:
            return self.build_output_fast(pert_idx, ctrl_idx, key)
        else:
            return self.build_output(pert_idx, ctrl_idx, key[0], key[1])

    # ---------------------------------------------------------
    # Sample a set (multiple cells, optimized version)
    # ---------------------------------------------------------
    def get_set_sample(self):
        key = self.unique_keys[random.randint(0, self._n_unique_keys - 1)]

        pert_arr = self.index_by_cov_pert[key]
        ctrl_arr = self.index_by_cov_ctrl[key[0]]
        
        n_pert = len(pert_arr)
        n_ctrl = len(ctrl_arr)
        cell_set_len = self.cell_set_len

        # Optimization: Use faster method if replacement is not needed
        if cell_set_len <= n_pert:
            pert_idxs = np.random.choice(pert_arr, size=cell_set_len, replace=False)
        else:
            pert_idxs = np.random.choice(pert_arr, size=cell_set_len, replace=True)
            
        if cell_set_len <= n_ctrl:
            ctrl_idxs = np.random.choice(ctrl_arr, size=cell_set_len, replace=False)
        else:
            ctrl_idxs = np.random.choice(ctrl_arr, size=cell_set_len, replace=True)

        if self.use_mix_pert:
            return self.build_output_fast(pert_idxs, ctrl_idxs, key)
        else:
            return self.build_output(pert_idxs, ctrl_idxs, key[0], key[1])

    # ---------------------------------------------------------
    # Fast build output (skip transform, used in mix_pert mode)
    # ---------------------------------------------------------
    def build_output_fast(self, pert_idx, ctrl_idx, key):
        """
        Ultra-optimized version:
        - Use pre-cached cov split results
        - Use torch.from_numpy (zero copy)
        - Skip transform, directly build output
        """
        cov = key[0]
        gene_pert, drug_pert, env_pert, crispr_type = key[1], key[2], key[3], key[4]
        
        # Use pre-cached transform
        t = self.transform
        
        # Pre-cached null embeddings
        gene_null = t._gene_null_emb
        drug_null = t._drug_null_emb
        env_null = t._env_null_emb
        crispr_null = t._crispr_null_emb
        
        out = {}
        
        # Use pre-split cov
        covs = self._cov_split_cache[cov]
        
        # Covariate processing
        if t.use_covs:
            cov_maps = t.cov_maps
            cov_null_embs = t._cov_null_embs
            for idx, cov_key in enumerate(self.cov_keys):
                out[cov_key] = cov_maps[cov_key].get(covs[idx], cov_null_embs[cov_key])
        
        # pert names
        if t.keep_pert_names:
            out['gene_names'] = gene_pert.split(t.comb_delim)
            out['drug_names'] = drug_pert.split(t.comb_delim)
            out['env_names'] = env_pert.split(t.comb_delim)
        
        # gene embedding (padded tensor format)
        if t.gene_pert_dim > 1:
            gene_map = t.gene_map
            gene_perts = gene_pert.split(t.comb_delim)
            max_gene = t.max_gene_perts
            n_gene = min(len(gene_perts), max_gene)
            gene_tensor = torch.zeros(max_gene, t.gene_pert_dim, dtype=torch.float32)
            for i in range(n_gene):
                gene_tensor[i] = gene_map.get(gene_perts[i], gene_null)
            out['gene_pert'] = gene_tensor
            out['gene_pert_len'] = n_gene
        
        # drug embedding (padded tensor format)
        if t.drug_pert_dim > 1:
            drug_map = t.drug_map
            drug_perts = drug_pert.split(t.comb_delim)
            max_drug = t.max_drug_perts
            n_drug = min(len(drug_perts), max_drug)
            drug_tensor = torch.zeros(max_drug, t.drug_pert_dim, dtype=torch.float32)
            for i in range(n_drug):
                drug_tensor[i] = drug_map.get(drug_perts[i], drug_null)
            out['drug_pert'] = drug_tensor
            out['drug_pert_len'] = n_drug
        
        # env embedding
        if t.env_pert_dim > 1:
            env_map = t.env_map
            env_perts = env_pert.split(t.comb_delim)
            if len(env_perts) == 1:
                out['env_pert'] = env_map.get(env_perts[0], env_null)
            else:
                result = env_map.get(env_perts[0], env_null).clone()
                for ep in env_perts[1:]:
                    result += env_map.get(ep, env_null)
                out['env_pert'] = result
        
        # crispr embedding
        if t.crispr_pert_dim > 1:
            out['crispr_pert'] = t.crispr_map.get(crispr_type, crispr_null)
        
        # counts (using torch.from_numpy zero copy, data is already float32 contiguous array)
        out['pert_cell_counts'] = torch.from_numpy(self.X_pert[pert_idx])
        out['control_cell_counts'] = torch.from_numpy(self.X_ctrl[ctrl_idx])
        
        # mask
        if self.pert_expression_mask is not None:
            out['mask'] = torch.from_numpy(self.pert_expression_mask[pert_idx])
        
        # cell embedding (data already converted to float32 contiguous array in __init__)
        if t.use_cell_emb and self.emb_pert is not None:
            out['pert_cell_emb'] = torch.from_numpy(self.emb_pert[pert_idx])
            out['control_cell_emb'] = torch.from_numpy(self.emb_ctrl[ctrl_idx])
        
        # raw counts
        if self.raw_pert is not None:
            out['pert_raw_counts'] = torch.as_tensor(self.raw_pert[pert_idx], dtype=torch.float32)
            out['control_raw_counts'] = torch.as_tensor(self.raw_ctrl[ctrl_idx], dtype=torch.float32)
        
        return out

    # ---------------------------------------------------------
    # Build transform input dictionary (original version, used in non mix_pert mode)
    # ---------------------------------------------------------
    def build_output(self, pert_idx, ctrl_idx, cov, pert):
        out = {}

        covs = self._cov_split_cache.get(cov) or cov.split('<>')
        for idx, cov_key in enumerate(self.cov_keys):
            out[cov_key] = covs[idx]

        if self.use_mix_pert:
            gene_pert, drug_pert, env_pert, crispr_type = pert
            out[self.gene_key] = gene_pert
            out[self.drug_key] = drug_pert
            out[self.env_key] = env_pert
            out[self.crispr_key] = crispr_type
        else:
            out[self.pert_key] = pert

        # counts (using torch.from_numpy)
        out["pert_cell_counts"] = torch.from_numpy(self.X_pert[pert_idx])
        out["control_cell_counts"] = torch.from_numpy(self.X_ctrl[ctrl_idx])
        
        # mask
        if self.pert_expression_mask is not None:
            out['mask'] = torch.from_numpy(self.pert_expression_mask[pert_idx])
            
        # embedding (data already converted to float32 contiguous array in __init__)
        if self.emb_pert is not None:
            out["pert_cell_emb"] = torch.from_numpy(self.emb_pert[pert_idx])
            out["control_cell_emb"] = torch.from_numpy(self.emb_ctrl[ctrl_idx])

        # raw
        if self.raw_pert is not None:
            out["pert_raw_counts"] = torch.as_tensor(self.raw_pert[pert_idx], dtype=torch.float32)
            out["control_raw_counts"] = torch.as_tensor(self.raw_ctrl[ctrl_idx], dtype=torch.float32)

        # transform (maintain compatibility)
        return self.transform(out)

    def get_gene_names(self):
        return self.pert_adata.var_names

    def get_embedding_width(self):
        if self.embedding_key:
            return self.pert_adata.obsm[self.embedding_key].shape[1]
        else:
            return None