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
from .datasets import  scTrainDataset,scInferenceDataset
from scipy.sparse import issparse,csr_matrix
from .collate import inference_collate,train_collate
from .utils import build_co_expression_graph

# Global cache: used to cache loaded adata objects to avoid repeated reading
_adata_cache: Dict[str, ad.AnnData] = {}

class PertDataModule(L.LightningDataModule):
    def __init__(self,
                 data_path,
                 train_batch_size:int,
                 val_batch_size:int,
                 test_batch_size:int,
                 control_val: str,
                 cov_keys: List[str],
                 mask_type: str='cell',
                 transform: DictConfig|None=None,
                 splitter: DictConfig|None = None, 
                 pert_key: str|None=None,
                 gene_key: str | None=None,
                 drug_key:str|None=None,
                 env_key:str|None=None,
                 crispr_key:str|None=None,#CRISPRi,CRISPRa,CRISPRko,null_token
                 result_avg_keys: List[str] | None = None,  # For example, if this is [celltype], the final result will be averaged over celltype during aggregation
                 evaluation: DictConfig | None =None,
                 perturbation_combination_delimiter: str | None=None,
                 embedding_key: str | None = None,
                 raw_counts_key: str | None = None,
                 cov_avg_sampling: bool = False,
                 sample_mode: str = "cell",  # "cell" | "set" - determines data packaging mode
                 cell_set_len: int | None = None,  # Only used when sample_mode="set"
                 add_keys: List[str] | None = None,
                 train_num_workers: int | None = 12,
                 val_num_workers: int | None = 12,
                 test_num_workers: int | None = 12,
                 co_expression_graph_config: DictConfig|None=None,
                 inference_top_hvg: int|None=None,
                 sharing_controls: bool=False,
                 predict_controls: bool=False,
                 task: str | None = None,  # Task name: unseen_cell, single_dataset_test, single_dataset_two_cellline, two_datasets_two_celllines
                 split_dir: str | None = None,  # Base directory for split CSV files
                 cache_data: bool = True,  # Whether to enable data caching (pin to memory)
                 **kwargs
                 ):

        super().__init__()
        self.use_mix_pert = pert_key is None and (gene_key is not None and drug_key is not None and env_key is not None)
        assert self.use_mix_pert or \
               (pert_key is not None and (gene_key is None and drug_key is None and env_key is None))

        self.result_avg_keys = result_avg_keys
        if self.result_avg_keys is None:
            self.result_avg_keys =cov_keys

        # Use caching mechanism: if data is already in memory, use it directly; otherwise load and cache
        # Note: The original data is cached, and each instance will create a copy based on the cached data and apply its own task/split processing
        data_path_str = str(Path(data_path).resolve())
        if cache_data and data_path_str in _adata_cache:
            print(f"Using cached adata for {data_path_str} (shape: {_adata_cache[data_path_str].shape})")
            # Use deep copy to avoid mutual influence between multiple instances
            self.adata = _adata_cache[data_path_str].copy()
        else:
            print(f"Loading adata from {data_path_str}...")
            self.adata = sc.read_h5ad(data_path)
            # Cache original data (before applying task/split processing)
            if cache_data:
                _adata_cache[data_path_str] = self.adata.copy()
                print(f"Cached adata for {data_path_str} (shape: {self.adata.shape})")
        
        # Read data split table
        # If task is specified, read split information from external CSV file and filter cells
        # If no task is specified, skip CSV matching and use data directly from data_path
        if task is not None:
            if split_dir is None:
                # Default split directory is the split folder under the project root
                # Try to infer project root from current file location
                current_file = Path(__file__)
                # modules.py is under src/VCBench/data/, go up 4 levels to project root
                project_root = current_file.parent.parent.parent.parent
                split_dir = project_root / "split"
                # If the inferred path doesn't exist, use absolute path
                if not split_dir.exists():
                    split_dir = Path("./split")
            else:
                split_dir = Path(split_dir)
            
            # Determine CSV file path based on task
            # Task corresponds to subdirectory name under split
            task_lower = str(task).lower()
            split_subdir = split_dir / task_lower

            def _choose_csv(path: Path) -> Path:
                if path.is_file():
                    return path
                # If multiple files, pick the first sorted for determinism
                csvs = sorted([p for p in path.glob("*.csv") if p.is_file()])
                if not csvs:
                    raise FileNotFoundError(f"No CSV found under {path}")
                return csvs[0]

            # Select CSV file (all tasks use the same logic)
            csv_path = _choose_csv(split_subdir)
            
            if not csv_path.exists():
                raise FileNotFoundError(f"Split CSV file not found: {csv_path}")
            
            print(f"Loading split CSV from: {csv_path}")
            # Read CSV, ensure split column is string type, not Categorical
            split_df = pd.read_csv(csv_path, index_col=0)
            # Remove possible spaces or invisible characters at both ends of column names (e.g., ' split', 'split ')
            split_df.columns = split_df.columns.astype(str).str.strip()

            if 'split' not in split_df.columns:
                raise KeyError(
                    f"'split' column not found in CSV {csv_path}. "
                    f"Available columns: {list(split_df.columns)}"
                )
            
            # Convert split column to string type (handle possible NaN or empty strings)
            split_df['split'] = split_df['split'].astype(str)
            # Replace empty strings and 'nan' strings with empty strings, then filter out
            split_df['split'] = split_df['split'].replace(['nan', 'NaN', ''], '')
            
            # Ensure CSV index (cell ID) matches adata index
            valid_split_mask = split_df['split'].isin(['train', 'val', 'test'])
            valid_cells = split_df[valid_split_mask].index
            
            # Check which cells exist in adata
            cells_in_adata = self.adata.obs.index.isin(valid_cells)
            n_matched = cells_in_adata.sum()
            n_total_valid = len(valid_cells)
            
            print(f"  Found {n_matched:,} cells in adata matching CSV (out of {n_total_valid:,} valid cells in CSV)")
            print(f"  Original adata shape: {self.adata.shape}")
            
            # Only keep cells with valid split values in CSV
            self.adata = self.adata[cells_in_adata].copy()
            
            # If original split column exists, delete it first to avoid type conflict
            if 'split' in self.adata.obs.columns:
                del self.adata.obs['split']
            
            split_series = split_df.loc[self.adata.obs.index, 'split']
            split_values = split_series.values.astype(str)
            split_series_new = pd.Series(split_values, index=self.adata.obs.index, dtype='object')
            self.adata._obs.loc[:, 'split'] = split_series_new
            self.adata._obs['split'] = self.adata._obs['split'].astype('object')

            assert not pd.api.types.is_categorical_dtype(self.adata.obs['split']), \
                "Failed to set split column as non-Categorical type"
            # If CSV contains sample_count, duplicate cells accordingly (with replacement)
            if 'sample_count' in split_df.columns:
                counts = pd.to_numeric(
                    split_df.loc[self.adata.obs.index, 'sample_count'],
                    errors='coerce'
                ).fillna(1).astype(int)
                counts[counts < 0] = 0  # negative counts are treated as zero
                repeat_idx = np.repeat(np.arange(self.adata.n_obs), counts.values)
                if len(repeat_idx) == 0:
                    raise ValueError("After applying sample_count, no cells remain to subset.")
                self.adata = self.adata[repeat_idx].copy()
                # Ensure unique obs names after duplication
                self.adata.obs_names_make_unique()

            # Drop cells whose expression is all zeros (sparse/dense safe)
            X = self.adata.X
            if issparse(X):
                row_sums = np.asarray(X.sum(axis=1)).ravel()
                col_sums = np.asarray(X.sum(axis=0)).ravel()
            else:
                row_sums = np.sum(X, axis=1)
                col_sums = np.sum(X, axis=0)

            # Drop cells with all-zero expression
            nonzero_mask = row_sums != 0
            n_drop = int((~nonzero_mask).sum())
            if n_drop > 0:
                print(f"  Dropping {n_drop:,} all-zero expression cells")
                self.adata = self.adata[nonzero_mask].copy()
                # recompute column sums after cell filtering
                X = self.adata.X
                if issparse(X):
                    col_sums = np.asarray(X.sum(axis=0)).ravel()
                else:
                    col_sums = np.sum(X, axis=0)

            # Drop genes (columns) that are all zero after subsetting
            gene_nonzero = col_sums != 0
            n_genes_drop = int((~gene_nonzero).sum())
            if n_genes_drop > 0:
                print(f"  Dropping {n_genes_drop:,} all-zero genes")
                self.adata = self.adata[:, gene_nonzero].copy()
        else:
            # If no task is provided, skip CSV matching and use data directly from data_path
            print("No task specified. Skipping split CSV matching. Using data directly from data_path.")
            # If the data already has a split column, keep it; if not, it will be generated by splitter later or an error will be raised


        # Store common attributes early for downstream use
        self.splitter = splitter
        self.pert_key = pert_key
        self.gene_key = gene_key
        self.drug_key=drug_key
        self.env_key=env_key
        self.control_val = control_val
        self.cov_keys = cov_keys
        self.crispr_key=crispr_key
        self.perturbation_combination_delimiter = perturbation_combination_delimiter
        # Guard against string "None"/"null" from CLI/Hydra serialization
        if embedding_key is None or str(embedding_key).lower() in ('none', 'null', ''):
            self.embedding_key = None
        else:
            self.embedding_key = embedding_key
        if raw_counts_key is None or str(raw_counts_key).lower() in ('none', 'null', ''):
            self.raw_counts_key = None
        else:
            self.raw_counts_key = raw_counts_key
        self.cov_avg_sampling = cov_avg_sampling
        self.sample_mode = sample_mode
        self.mask_type = mask_type
        # Only use cell_set_len when sample_mode is "set"
        if sample_mode == "set":
            self.cell_set_len = cell_set_len if cell_set_len is not None else 128  # Default to 128 for set mode
        else:
            self.cell_set_len = None  # Force None for cell mode

        self.add_keys = add_keys
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.test_num_workers = test_num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.evaluation = evaluation
        self.predict_controls = predict_controls
        if self.use_mix_pert:
            # Build merged perturbation column and robust control mask
            merged_pert_col = self.merge_cols(
                self.adata.obs,
                cols=[gene_key, drug_key, env_key, crispr_key],
            )

            def _col_to_str(col_name):
                if col_name is None or col_name not in self.adata.obs.columns:
                    return pd.Series([""] * self.adata.n_obs, index=self.adata.obs.index)
                col = self.adata.obs[col_name]
                if pd.api.types.is_categorical_dtype(col):
                    col = col.astype(str)
                return col.fillna("").astype(str)

            # Prefer an existing boolean control column if present; otherwise derive
            if control_val in self.adata.obs.columns and pd.api.types.is_bool_dtype(
                self.adata.obs[control_val]
            ):
                is_control = self.adata.obs[control_val].astype(bool).copy()
            else:
                g_col = _col_to_str(gene_key)
                d_col = _col_to_str(drug_key)
                e_col = _col_to_str(env_key)
                c_col = _col_to_str(crispr_key)

                def _is_control_col(series: pd.Series) -> pd.Series:
                    return (series == "") | (series == str(self.control_val))

                is_control = (
                    _is_control_col(g_col)
                    & _is_control_col(d_col)
                    & _is_control_col(e_col)
                    & _is_control_col(c_col)
                )

            merged_pert_col = pd.Series(merged_pert_col, index=self.adata.obs.index).astype(str)
            merged_pert_col.loc[is_control] = self.control_val
            self.adata.obs['_merged_pert_col_'] = merged_pert_col
            # Ensure a clean boolean control column for downstream splits
            self.adata.obs[self.control_val] = is_control.astype(bool)

        # Only enforce batch_size=1 for set mode (when cell_set_len is used)
        if self.sample_mode == "set" and self.cell_set_len:
            self.val_batch_size=1
            self.test_batch_size=1

        self.transform = transform

        if splitter:
            split_dict = datasplitter.PerturbationDataSplitter.split_dataset(
                splitter_config=splitter,
                obs_dataframe=self.adata.obs,
                perturbation_key=pert_key,
                perturbation_combination_delimiter=perturbation_combination_delimiter,
                perturbation_control_value=control_val,
            )
        else:
            # If no splitter is provided, try to read split column from obs
            # This may come from:
            # 1. External CSV loaded through task (already added to obs)
            # 2. Existing split column in the data file
            if 'split' not in self.adata.obs.columns:
                raise ValueError(
                    "'split' column not found in adata.obs. "
                    "Please provide one of the following:\n"
                    "  1. task parameter (to load split from CSV file)\n"
                    "  2. splitter config (to generate split automatically)\n"
                    "  3. split column in the input data file"
                )
            split = self.adata.obs['split']
            split_dict = {}
            split_dict['train'] = split == 'train'
            split_dict['val'] = split == 'val'
            split_dict['test'] = split == 'test'

        self.generate_mask_dict()

        '''Convert all to dense format and ensure float32 type'''
        self.adata.X = self.to_dense(self.adata.X).astype(np.float32)
        if self.embedding_key:
            self.adata.obsm[self.embedding_key] = self.to_dense(self.adata.obsm[self.embedding_key]).astype(np.float32)
        if self.raw_counts_key:
            self.adata.layers[self.raw_counts_key] = self.to_dense(self.adata.layers[self.raw_counts_key]).astype(np.float32)


        if inference_top_hvg and 'highly_variable_rank' in self.adata.var:
            self.inference_top_hvg=self.adata.var['highly_variable_rank'].\
                                       fillna(np.inf).values.argsort()[:inference_top_hvg]

        # In mix_pert mode, use _merged_pert_col_ instead of pert_key
        if self.use_mix_pert:
            sharing_control_adata=self.adata[self.adata.obs['_merged_pert_col_']==self.control_val]
        else:
            sharing_control_adata=self.adata[self.adata.obs[self.pert_key]==self.control_val]

        train_adata = self.adata[split_dict["train"]]
        train_pert_adata=train_adata[~train_adata.obs[control_val]]
        train_control_adata = train_adata[train_adata.obs[control_val]]
        if sharing_controls:
            train_control_adata=sharing_control_adata

        val_adata = self.adata[split_dict["val"]]
        val_pert_adata = val_adata[~val_adata.obs[control_val]]
        val_control_adata = val_adata[val_adata.obs[control_val]]
        if sharing_controls:
            val_control_adata=sharing_control_adata

        test_adata = self.adata[split_dict["test"]]
        test_pert_adata = test_adata[~test_adata.obs[control_val]]
        test_control_adata = test_adata[test_adata.obs[control_val] ]
        if sharing_controls:
            test_control_adata=sharing_control_adata

        if co_expression_graph_config:
            build_co_expression_graph(**co_expression_graph_config,control_adata=sharing_control_adata)


        '''Get dataset'''
        self.train_dataset =scTrainDataset(
            pert_adata=train_pert_adata,
            control_adata=train_control_adata,
            pert_key=self.pert_key,
            gene_key=self.gene_key,
            crispr_key=self.crispr_key,
            drug_key=self.drug_key,
            env_key=self.env_key,
            use_mix_pert=self.use_mix_pert,
            cov_keys=self.cov_keys,
            transform=self.transform,
            embedding_key=self.embedding_key,
            raw_counts_key=self.raw_counts_key,
            cov_avg_sampling=self.cov_avg_sampling,
            cell_set_len=self.cell_set_len,
            add_keys=self.add_keys,
            predict_controls=self.predict_controls,
            cellclass_mask_dict=self.cellclass_mask_dict,
            mask_type=self.mask_type,
            **kwargs
        )

        self.val_dataset=scInferenceDataset(
            pert_adata=val_pert_adata,
            control_adata=val_control_adata,
            pert_key=self.pert_key,
            gene_key=self.gene_key,
            crispr_key=self.crispr_key,
            drug_key=self.drug_key,
            env_key=self.env_key,
            use_mix_pert=self.use_mix_pert,
            cov_keys=self.cov_keys,
            transform=self.transform,
            embedding_key=self.embedding_key,
            raw_counts_key=self.raw_counts_key,
            cell_set_len=self.cell_set_len,
            add_keys=self.add_keys,
            cellclass_mask_dict=self.cellclass_mask_dict,
            mask_type=self.mask_type,
            **kwargs
        )

        self.test_dataset = scInferenceDataset(
            pert_adata=test_pert_adata,
            control_adata=test_control_adata,
            pert_key=self.pert_key,
            gene_key=self.gene_key,
            crispr_key=self.crispr_key,
            drug_key=self.drug_key,
            env_key=self.env_key,
            use_mix_pert=self.use_mix_pert,
            cov_keys=self.cov_keys,
            transform=self.transform,
            embedding_key=self.embedding_key,
            raw_counts_key=self.raw_counts_key,
            cell_set_len=self.cell_set_len,
            add_keys=self.add_keys,
            cellclass_mask_dict=self.cellclass_mask_dict,
            mask_type=self.mask_type,
            **kwargs
        )

        self.infer_collect_fn=inference_collate()
        self.train_collect_fn=train_collate()

        # Calculate cov_dim from the instantiated transform
        self.cov_dim = None
        if hasattr(self.train_dataset, 'transform') and hasattr(self.train_dataset.transform, 'cov_dims'):
            cov_dims = self.train_dataset.transform.cov_dims
            if cov_dims:
                self.cov_dim = sum(cov_dims.values())
        
    def train_dataloader(self) -> DataLoader:

        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.train_num_workers,
            collate_fn=self.train_collect_fn,
            drop_last=True,
            shuffle=False,
        )
    def val_dataloader(self) -> DataLoader | None:

        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.val_num_workers,
            collate_fn=self.infer_collect_fn,
            shuffle=False,
        )
    def test_dataloader(self) -> DataLoader | None:

        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.test_num_workers,
            shuffle=False,
            collate_fn=self.infer_collect_fn,
        )
    def to_dense(self,X):
        if isinstance(X, csr_matrix):
            return X.toarray()
        elif issparse(X):
            # For other types of sparse matrices (e.g., csc_matrix), can choose whether to convert
            return X.toarray()
        else:
            return X
    def merge_cols(self, obs_df, cols):
        # Empty cols case: return all empty string columns
        if cols is None or (isinstance(cols, list) and len(cols) == 0):
            return np.full(len(obs_df), '', dtype=object)
        # Process first column, convert to string type first to avoid Categorical issues
        first_col = obs_df[cols[0]]
        # If it's Categorical, convert to string first
        if pd.api.types.is_categorical_dtype(first_col):
            first_col = first_col.astype(str)
        merged = first_col.fillna('').astype(str).to_numpy()

        for c in cols[1:]:
            # Process subsequent columns, convert to string type first to avoid Categorical issues
            col_data = obs_df[c]
            # If it's Categorical, convert to string first
            if pd.api.types.is_categorical_dtype(col_data):
                col_data = col_data.astype(str)
            col_values = col_data.fillna('').astype(str).to_numpy()
            merged = merged + "<>" + col_values
        return merged
    def generate_mask_dict(self):
        self.adata.obs['cellclass'] =self.merge_cols(self.adata.obs, self.result_avg_keys)
        mask_dict = {}
        for cellclass_key in self.adata.obs['cellclass'].unique():
            subset_data = self.adata.X[self.adata.obs['cellclass']==cellclass_key]
            # Ensure dense format before summing
            if issparse(subset_data):
                subset_data = subset_data.toarray()
            mask_dict[cellclass_key] = np.asarray(subset_data).sum(axis=0).flatten() != 0
        self.cellclass_mask_dict = mask_dict