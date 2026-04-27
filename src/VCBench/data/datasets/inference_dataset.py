import warnings
from pathlib import Path
from collections.abc import Callable, Mapping
import gc
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from omegaconf import DictConfig
import lightning as L
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import VCBench.data.datasplitter as datasplitter
import anndata as ad
import numpy as np
import hydra
import scanpy as sc
import pandas as pd
import torch
import random  # Python random is faster than np.random
from scipy.sparse import issparse


class scInferenceDataset(Dataset):

    def __init__(
        self,
        pert_adata: ad.AnnData,
        control_adata: ad.AnnData,
        transform: DictConfig,
        pert_key: str | None = None,
        gene_key: str | None = None,
        drug_key: str | None = None,
        env_key: str | None = None,
        crispr_key: str | None = None,
        use_mix_pert: bool = False,
        cov_keys: List[str] | None = None,
        embedding_key: str | None = None,
        raw_counts_key: str | None = None,
        cell_set_len: int | None = None,
        add_keys: List[str] | None = None,
        cellclass_mask_dict: Dict[str, np.ndarray] | None = None,
        mask_type: str | None = None,
        **kwargs
    ):
        super().__init__()

        self.merge_delim = '<>'
        self.pert_key = pert_key
        self.gene_key = gene_key
        self.drug_key = drug_key
        self.env_key = env_key
        self.crispr_key = crispr_key
        self.cov_keys = cov_keys
        self.add_keys = add_keys
        self.embedding_key = embedding_key
        self.raw_counts_key = raw_counts_key
        self.cell_set_len = cell_set_len
        self.mask_type = mask_type

        self.use_mix_pert = use_mix_pert

        # ====== 1) Prefetch obs-level information (avoid repeated pandas operations) ======
        pert_obs = pert_adata.obs.copy()
        ctrl_obs = control_adata.obs.copy()

        pert_obs["cov_merged"] = self.merge_cols(pert_obs, cov_keys, self.merge_delim)
        ctrl_obs["cov_merged"] = self.merge_cols(ctrl_obs, cov_keys, self.merge_delim)

        self.pert_adata = pert_adata
        self.control_adata = control_adata

        self.pert_obs = pert_obs
        self.control_obs = ctrl_obs

        # ====== 2) Prebuild control group index mapping ======
        # Use integer indices instead of string indices (faster)
        ctrl_cov_arr = ctrl_obs["cov_merged"].to_numpy()
        self.ctrl_group_map = {}
        for i, cov in enumerate(ctrl_cov_arr):
            self.ctrl_group_map.setdefault(cov, []).append(i)
        # Convert to numpy array (speed up random.choice)
        self.ctrl_group_map = {k: np.array(v, dtype=np.int64) for k, v in self.ctrl_group_map.items()}
        # Save all control indices as fallback
        self.all_ctrl_idxs = np.arange(len(ctrl_obs), dtype=np.int64)

        # ====== 3) Extract matrix to numpy at once, ensure dense + float32 + contiguous ======
        X_pert = pert_adata.X
        X_ctrl = control_adata.X
        self.pert_X = X_pert.toarray() if issparse(X_pert) else np.asarray(X_pert)
        self.ctrl_X = X_ctrl.toarray() if issparse(X_ctrl) else np.asarray(X_ctrl)
        
        # Convert to float32 contiguous array (speed up tensor conversion)
        if self.pert_X.dtype != np.float32:
            self.pert_X = self.pert_X.astype(np.float32)
        if self.ctrl_X.dtype != np.float32:
            self.ctrl_X = self.ctrl_X.astype(np.float32)
        self.pert_X = np.ascontiguousarray(self.pert_X)
        self.ctrl_X = np.ascontiguousarray(self.ctrl_X)

        # Mask processing
        if self.mask_type == 'cell':
            self.pert_expression_mask = np.ascontiguousarray((self.pert_X > 0).astype(np.float32))
        elif self.mask_type is not None and cellclass_mask_dict is not None:
            mask = [cellclass_mask_dict[cellclass] for cellclass in self.pert_adata.obs['cellclass']]
            self.pert_expression_mask = np.ascontiguousarray(np.stack(mask, axis=0), dtype=np.float32)
        else:
            self.pert_expression_mask = None

        # Convert embedding to float32 contiguous array
        if embedding_key:
            self.pert_emb = np.ascontiguousarray(pert_adata.obsm[embedding_key], dtype=np.float32)
            self.ctrl_emb = np.ascontiguousarray(control_adata.obsm[embedding_key], dtype=np.float32)
        else:
            self.pert_emb = self.ctrl_emb = None

        if raw_counts_key:
            self.pert_raw = np.ascontiguousarray(pert_obs[raw_counts_key].to_numpy(), dtype=np.float32)
            self.ctrl_raw = np.ascontiguousarray(ctrl_obs[raw_counts_key].to_numpy(), dtype=np.float32)
        else:
            self.pert_raw = self.ctrl_raw = None

        # Integer index mapping (no need for string mapping anymore)
        self.pert_index_map = {name: i for i, name in enumerate(pert_obs.index)}
        self.ctrl_index_map = {name: i for i, name in enumerate(ctrl_obs.index)}

        # ====== Initialize transform ======
        self.transform = transform(obs_df=pert_obs.copy())

        # ====== Build chunks ======
        if cell_set_len:
            self.chunks = self.build_sets()
        else:
            self.chunks = self.build_singles()

    # ----------------------------------------------------------------------
    # Fast packaging of expression information (zero-copy version)
    # ----------------------------------------------------------------------
    def pack_expr(self, pert_i, ctrl_i):
        """Use torch.from_numpy to achieve zero copy, data has been converted to float32 contiguous array in __init__"""
        out = {
            "pert_cell_counts": torch.from_numpy(self.pert_X[pert_i]),
            "control_cell_counts": torch.from_numpy(self.ctrl_X[ctrl_i]),
        }

        # expression mask for loss calculation
        if self.pert_expression_mask is not None:
            out["mask"] = torch.from_numpy(self.pert_expression_mask[pert_i])

        if self.pert_emb is not None:
            out["pert_cell_emb"] = torch.from_numpy(self.pert_emb[pert_i])
            out["control_cell_emb"] = torch.from_numpy(self.ctrl_emb[ctrl_i])

        if self.pert_raw is not None:
            out["pert_raw_counts"] = torch.from_numpy(self.pert_raw[pert_i])
            out["control_raw_counts"] = torch.from_numpy(self.ctrl_raw[ctrl_i])

        return out

    # ----------------------------------------------------------------------
    def build_singles(self):
        """Optimized version: pre-extract all columns to numpy, avoid pandas operations in loop"""
        chunks = []
        n = len(self.pert_obs)

        # Get all meta information as numpy arrays at once
        pert_cov = self.pert_obs["cov_merged"].to_numpy()
        
        # Pre-extract cov columns
        cov_arrays = {c: self.pert_obs[c].to_numpy() for c in self.cov_keys}
        
        if self.use_mix_pert:
            gene_pert = self.pert_obs[self.gene_key].to_numpy()
            drug_pert = self.pert_obs[self.drug_key].to_numpy()
            env_pert = self.pert_obs[self.env_key].to_numpy()
            crispr_type = self.pert_obs[self.crispr_key].to_numpy()
        else:
            pert_pert = self.pert_obs[self.pert_key].to_numpy()

        add_keys_arrays = {
            k: self.pert_obs[k].to_numpy() for k in (self.add_keys or [])
        }

        # Pre-cache transform
        transform = self.transform
        pert_obs = self.pert_obs

        for i in range(n):
            cov = pert_cov[i]
            ctrl_arr = self.ctrl_group_map.get(cov, self.all_ctrl_idxs)
            # Use Python random (faster)
            ctrl_i = ctrl_arr[random.randint(0, len(ctrl_arr) - 1)]

            # Use pre-extracted numpy arrays (avoid pandas iloc)
            meta = {c: cov_arrays[c][i] for c in self.cov_keys}
            if self.use_mix_pert:
                meta[self.gene_key] = gene_pert[i]
                meta[self.drug_key] = drug_pert[i]
                meta[self.env_key] = env_pert[i]
                meta[self.crispr_key] = crispr_type[i]
            else:
                meta[self.pert_key] = pert_pert[i]
            for k in add_keys_arrays:
                meta[k] = add_keys_arrays[k][i]

            expr = self.pack_expr(i, ctrl_i)
            chunks.append((transform({**meta, **expr}), pert_obs.iloc[[i]]))

        return chunks

    # ----------------------------------------------------------------------
    def build_sets(self):
        """Optimized version: reduce calculations in loop"""
        chunks = []
        merge_delim = self.merge_delim
        
        if self.use_mix_pert:
            obs_merged = self.merge_cols(
                self.pert_obs, self.cov_keys + [self.gene_key, self.drug_key, self.env_key, self.crispr_key], merge_delim
            ).to_numpy()
        else:
            obs_merged = self.merge_cols(
                self.pert_obs, self.cov_keys + [self.pert_key], merge_delim
            ).to_numpy()

        unique_groups = np.unique(obs_merged)
        n_cov_keys = len(self.cov_keys)
        
        # Pre-extract add_keys arrays
        add_keys_arrays = {k: self.pert_obs[k].to_numpy() for k in (self.add_keys or [])}
        
        # Pre-cache
        transform = self.transform
        pert_obs = self.pert_obs
        cell_set_len = self.cell_set_len

        for g in unique_groups:
            idxs = np.where(obs_merged == g)[0]
            cov_vals = g.split(merge_delim)
            
            if self.use_mix_pert:
                pert_val = cov_vals[n_cov_keys:]
                cov_vals = cov_vals[:n_cov_keys]
                gene_pert, drug_pert, env_pert, crispr_type = pert_val
                sample_meta = {
                    self.gene_key: gene_pert,
                    self.drug_key: drug_pert,
                    self.env_key: env_pert,
                    self.crispr_key: crispr_type,
                }
            else:
                pert_val = cov_vals[-1]
                cov_vals = cov_vals[:-1]
                sample_meta = {self.pert_key: pert_val}

            for ci, c in enumerate(self.cov_keys):
                sample_meta[c] = cov_vals[ci]

            ctrl_cov = merge_delim.join(cov_vals)
            ctrl_idxs = self.ctrl_group_map.get(ctrl_cov, self.all_ctrl_idxs)
            n_ctrl = len(ctrl_idxs)

            for start in range(0, len(idxs), cell_set_len):
                sub = idxs[start:start + cell_set_len]
                sub_n = len(sub)

                # Directly use integer indices (ctrl_idxs is already an integer array)
                ctrl_sample = np.random.choice(ctrl_idxs, size=sub_n, replace=(n_ctrl < sub_n))

                # Pack batch (ctrl_sample is already an integer index)
                out = self.pack_expr(sub, ctrl_sample)

                # add_keys (use pre-extracted arrays)
                for k in add_keys_arrays:
                    out[k] = add_keys_arrays[k][sub]

                chunks.append((transform({**sample_meta, **out}), pert_obs.iloc[sub]))

        return chunks

    # ----------------------------------------------------------------------

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]

    @staticmethod
    def merge_cols(df, cols, delim):
        x = df[cols[0]].astype(str)
        for c in cols[1:]:
            x = x + delim + df[c].astype(str)
        return x.astype("category")