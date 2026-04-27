from __future__ import annotations
import logging

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from pandas.api.types import CategoricalDtype

import torch
from torch.utils.data import (
    DataLoader,
    BatchSampler,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)

from .types import FrozenDictKeyMap
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from  multiprocessing import Pool

log = logging.getLogger(__name__)


def unique_perturbations(p: pd.Series, combo_delim: str = "+"):
    """Return unique perturbations from a pandas Series"""
    unique_perts = set()
    for pert in p.unique():
        unique_perts.update(pert.split(combo_delim))
    return list(unique_perts)


def parse_perturbation_combinations(
    combined_perturbations: pd.Series,
    delimiter: str | None = "+",
    perturbation_control_value: str | None = "control",
) -> tuple[ArrayLike, list[str]]:
    """Get all perturbations applied to each cell.

    Args:
        combined_perturbations: combined perturbations string representation of
          size (n_cells, )
        delimiter: a string that separates individual perturbations
        perturbation_control_value: a string that represents the control perturbation

    Returns:
        A tuple (combinations, unique_perturbations), where combinations is an
          (n_cell, ) array of lists of individual perturbations applied to each
          cell, and unique_perturbations is a set of all unique perturbations.
    """
    assert isinstance(combined_perturbations.dtype, CategoricalDtype)

    # Split the perturbations by the delimiter
    parsed = []
    uniques = {}  ## Store unique perturbations as dictionary keys to ensure ordering is the same
    for combination in combined_perturbations.astype(str):
        perturbation_list = []
        for perturbation in combination.split(delimiter):
            if perturbation != perturbation_control_value:
                perturbation_list.append(perturbation)
                uniques[perturbation] = None
        parsed.append(perturbation_list)

    uniques = list(uniques.keys())
    return parsed, uniques


def restore_perturbation_combinations(
    parsed_perturbations: list[list[str]],
    delimiter: str | None = "+",
    perturbation_control_value: str | None = "control",
) -> pd.Series:
    """Restore the combined perturbations from a list of perturbations using a specified delimiter and control value

    Args:
        parsed_perturbations: a list of lists of perturbations
        delimiter: a string that separates individual perturbations
        perturbation_control_value: a string that represents the control perturbation

    Returns:
        A pandas Series of combined perturbations
    """
    combined_perturbations = []
    for combined_perts in parsed_perturbations:
        assert isinstance(combined_perts, list)
        pert_joined = delimiter.join(combined_perts)
        if pert_joined == "":
            pert = perturbation_control_value
        else:
            pert = pert_joined
        combined_perturbations.append(pert)

    combined_perturbations = pd.Series(combined_perturbations, dtype="category")
    return combined_perturbations


def get_covariates(df: pd.DataFrame, covariate_keys: list[str]):
    """Get covariates from a dataframe.

    Args:
        df: a dataframe containing covariates for each cell with n_cells rows
        covariate_keys: a list of covariate keys in the dataframe

    Returns:
        A tuple (covariates, covariate_unique_values), where covariates is a
          dictionary of covariate keys to covariate values with n_cells rows,
          and covariate_unique_values is a dictionary of covariate keys to
          unique covariate values.

    Raises:
        KeyError: if a covariate key is not found in the dataframe.
    """
    try:
        covariates: np.ndarray = df[covariate_keys].values
        covariate_unique_values = {
            cov: list(df[cov].unique()) for cov in covariate_keys
        }
    except KeyError as e:
        raise KeyError("Covariate key not found in dataframe: " + str(e)) from e
    return dict(zip(covariate_keys, covariates.T)), covariate_unique_values


def build_control_dict(
    control_covariate_df: pd.DataFrame,
    covariate_keys: list[str] | None = None,
):
    """Build a dictionary of controls for each covariate condition.

    Args:
        control_covariate_df: pandas dataframe containing the covariates for each sample/cell
        covariate_keys: a list of keys in adata.obs that contain the covariates

    Returns:
        A dictionary of controls, where each control is a sparse matrix of size
          (n_controls_per_condition, n_genes) and the keys are
          dict[covariate_key, covariate_value].
    """
    if covariate_keys is None:
        covariate_keys = control_covariate_df.columns.tolist()

    grouped = control_covariate_df.groupby(
        list(covariate_keys)
    )  # groupby requires a list
    control_indexes = FrozenDictKeyMap()
    for group_key, group_indices in grouped.indices.items():
        if len(covariate_keys) == 1:
            assert isinstance(group_key, (str, int))
            group_key = (group_key,)

        key = dict(zip(covariate_keys, group_key))
        control_indexes[key] = group_indices

    return control_indexes


def batch_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool = True,
    oversample: bool = False,
    oversample_root: float = 2.0,
    **kwargs,
):
    """
    Build a PyTorch DataLoader from a PyTorch Dataset using a BatchSampler.

    Args:
        dataset: a PyTorch Dataset
        batch_size: the batch size
        shuffle: whether to shuffle the data
        oversample: whether to oversample the data
        oversample_root: oversampling weight will be
          `(1/class_frac)^(1/oversample_root)`
        kwargs: additional arguments to pass to DataLoaders
    """
    if oversample:
        assert oversample_root > 0, "Oversample root must be greater than 0"

        weights_dictionary = {}
        covariates_df = pd.DataFrame(dataset.covariates)
        covariates_df["concatenated"] = covariates_df.apply(
            lambda row: "_".join(row.values.astype(str)), axis=1
        )
        covariate_fractions = (
            covariates_df["concatenated"].value_counts() / covariates_df.shape[0]
        )
        for covariate, frac in covariate_fractions.items():
            weight = (1 / frac) ** (1.0 / oversample_root)
            weights_dictionary[covariate] = weight

        weights = []
        for i in range(0, len(dataset)):
            cov_values = [values[i] for values in dataset.covariates.values()]
            cov_key = "_".join(cov_values)
            weight = weights_dictionary[cov_key]
            weights.append(weight)

        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)

    elif shuffle:
        sampler = RandomSampler(dataset)

    else:
        sampler = SequentialSampler(dataset)

    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    dataloader = DataLoader(dataset, sampler=batch_sampler, **kwargs)
    return dataloader

class GraphBuilder:
    def __init__(self,data_dir,pert_key,control_val,comb_delim='+',num_workers=25):
        self.k=None
        self.threshold=None
        self.data_dir=data_dir
        self.num_workers=num_workers
        self.pert_key=pert_key
        self.control_val=control_val
        self.comb_delim=comb_delim
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def get_GO_edge_list(self,args):
        """
        Get gene ontology edge list
        """
        g1, gene2go = args
        edge_list = []
        for g2 in gene2go.keys():
            score = len(gene2go[g1].intersection(gene2go[g2])) / len(
                gene2go[g1].union(gene2go[g2]))
            if score > self.threshold:
                edge_list.append((g2, g1, score))
                edge_list.sort(key=lambda x: -x[2])
        return edge_list[:self.k+1]

    def make_GO(self,gene2go):
        """
        Creates Gene Ontology graph from a custom set of genes
        """
        fname = os.path.join(self.data_dir, "go.csv")
        if os.path.exists(fname):
            return pd.read_csv(fname)


        print('Creating custom GO graph, this can take a few minutes')

        with Pool(self.num_workers) as p:
            all_edge_list = list(
                tqdm(p.imap(self.get_GO_edge_list, ((g, gene2go) for g in gene2go.keys())),
                     total=len(gene2go.keys())))
        edge_list = []
        print('Concat edge list')
        for i in tqdm(all_edge_list):
            edge_list = edge_list + i

        df_out = pd.DataFrame(edge_list).rename(
            columns={0: 'source', 1: 'target', 2: 'importance'})

        print('Saving GO_df to file')
        df_out.to_csv(fname, index=False)
        print('Done!')

        return df_out

    def get_similarity_network(self,network_type,control_adata, threshold, k,
                               gene2go):
        self.k=k
        self.threshold=threshold
        if network_type == 'co-express':
            df_out = self.get_coexpression_network_from_train(control_adata)
        elif network_type == 'go':
            df_out = self.make_GO(gene2go)

        return df_out

    def get_coexpression_network_from_train(self,control_adata):

        fname = os.path.join(self.data_dir, "co-express.csv")
        if os.path.exists(fname):
            return pd.read_csv(fname)

        threshold=self.threshold
        adata=control_adata
        k=self.k
        if os.path.exists(fname):
            return pd.read_csv(fname)
        else:
            gene_list = [f for f in adata.var_names.values]
            idx2gene = dict(zip(range(len(gene_list)), gene_list))
            X_tr = adata.X

            from scipy.sparse import issparse
            if issparse(X_tr):
                X_tr = X_tr.toarray()

            out = np_pearson_cor(X_tr, X_tr)
            out[np.isnan(out)] = 0
            out = np.abs(out)

            out_sort_idx = np.argsort(out)[:, -(k + 1):]
            out_sort_val = np.sort(out)[:, -(k + 1):]

            df_g = []
            for i in range(out_sort_idx.shape[0]):
                target = idx2gene[i]
                for j in range(out_sort_idx.shape[1]):
                    df_g.append((idx2gene[out_sort_idx[i, j]], target, out_sort_val[i, j]))

            df_g = [i for i in df_g if i[2] > threshold]
            df_co_expression = pd.DataFrame(df_g).rename(columns={0: 'source',
                                                                  1: 'target',
                                                                  2: 'importance'})
            df_co_expression.to_csv(fname, index=False)
            return df_co_expression


def np_pearson_cor(x, y):
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)

def build_co_expression_graph(dir,control_adata,topk,threshold,**kwargs):
    from scipy.sparse import csr_matrix

    fname = os.path.join(dir, 'co-expression_graph.pt')
    gene2idx_path = os.path.join(dir, 'gene2idx.pt')
    threshold=threshold
    adata=control_adata
    k=topk

    if os.path.exists(fname):
        gene2idx = torch.load(gene2idx_path,weights_only=False)
        return torch.load(fname,weights_only=False),gene2idx
    else:
        X = adata.X
        X_tr = X
        X_tr = X_tr.toarray()
        out = np_pearson_cor(X_tr, X_tr)
        out[np.isnan(out)] = 0
        out = np.abs(out)

        out_sort_idx = np.argsort(out)[:, -(k + 1):]
        out_sort_val = np.sort(out)[:, -(k + 1):]

        df_g = []
        for i in range(out_sort_idx.shape[0]):
            target = i
            for j in range(out_sort_idx.shape[1]):
                df_g.append((out_sort_idx[i, j], target, out_sort_val[i, j]))

        df_g = [i for i in df_g if i[2] > threshold]
        df_co_expression = pd.DataFrame(df_g).rename(columns={0: 'source',
                                                              1: 'target',
                                                              2: 'importance'})

        source=df_co_expression['source']
        target=df_co_expression['target']
        weight=df_co_expression['importance']

        n_genes=len(adata.var_names)

        adj_M=csr_matrix((weight,(source,target)),shape=(n_genes,n_genes))

        gene2idx = {}
        gene_list = adata.var_names
        for idx, gene_name in enumerate(gene_list):
            gene2idx[gene_name] = idx

        torch.save(gene2idx, gene2idx_path)
        torch.save(adj_M, fname)

        return adj_M,gene2idx


