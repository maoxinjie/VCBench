from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from scipy.stats import pearsonr, spearmanr
import numpy as np
from numpy.linalg import norm
import pandas as pd
import tqdm
from ..utils import merge_cols


def compute_metric(x, y, metric):
    """Compute specified similarity/distance metric between x and y vectors"""

    if metric == "pearson":
        score = pearsonr(x, y)[0]
    elif metric == 'spearman':
         score = spearmanr(x, y)[0]
    elif metric == "r2_score":
        score = r2_score(x, y)
    elif metric == "cosine":
        score = np.dot(x, y) / (norm(x) * norm(y))
    elif metric == "mse":
        score = np.mean(np.square(x - y))
    elif metric == "rmse":
        score = np.sqrt(np.mean(np.square(x - y)))
    elif metric == "mae":
        score = np.mean(np.abs(x - y))

    return score


def compare_perts(
    pred, ref, features=None, perts=None, metric="pearson", deg_dict=None
):
    """Compare expression similarities between `pred` and `ref` DataFrames using the specified metric"""

    if perts is None:
        perts = list(set(pred.index).intersection(ref.index))
        assert len(perts) > 0
    else:
        perts = list(perts)

    if features is not None:
        # Only subset on features that exist in both pred/ref to avoid KeyError
        common_feats = [f for f in features if f in pred.columns and f in ref.columns]
        pred = pred.loc[:, common_feats]
        ref = ref.loc[:, common_feats]

    pred = pred.loc[perts, :]
    ref = ref.loc[perts, :]

    pred = pred.replace([np.inf, -np.inf], 0)
    ref = ref.replace([np.inf, -np.inf], 0)

    eval_metric = []
    for p in perts:
        if deg_dict is not None:
            # Only keep genes that exist in the current DataFrame to avoid KeyError due to masking/subsetting
            genes = [g for g in deg_dict[p] if g in ref.columns]
            if len(genes) == 0:
                # If all DEGs for this perturbation are filtered out in the current feature subset,
                # Fall back to using all available features
                genes = list(ref.columns)
        else:
            genes = ref.columns

        eval_metric.append(
            compute_metric(pred.loc[p, genes], ref.loc[p, genes], metric)
        )

    eval_scores = pd.Series(index=perts, data=eval_metric)
    return eval_scores


def pairwise_metric_helper(
    df,
    df2=None,
    metric="rmse",
    pairwise_deg_dict=None,
    verbose=False,
):
    if df2 is None:
        df2 = df

    mat = pd.DataFrame(0.0, index=df.index, columns=df2.index)
    for p1 in tqdm.tqdm(df.index, disable=not verbose):
        for p2 in df2.index:
            if pairwise_deg_dict is not None:
                pp = frozenset([p1, p2])
                genes_ix = pairwise_deg_dict[pp]

                m = compute_metric(
                    df.loc[p1, genes_ix],
                    df2.loc[p2, genes_ix],
                    metric=metric,
                )
            else:
                m = compute_metric(
                    df.loc[p1],
                    df2.loc[p2],
                    metric=metric,
                )
            mat.at[p1, p2] = m

    return mat


def rank_helper(pred_ref_mat, metric_type):
    rel_ranks = pd.Series(1.0, index=pred_ref_mat.columns)
    for p in pred_ref_mat.columns:
        pred_metrics = pred_ref_mat.loc[:, p]
        pred_metrics = pred_metrics.sample(frac=1.0)  ## Shuffle to avoid ties
        if metric_type == "distance":
            pred_metrics = pred_metrics.sort_values(ascending=True)
        elif metric_type == "similarity":
            pred_metrics = pred_metrics.sort_values(ascending=False)
        else:
            raise ValueError(
                "Invalid metric_type, should be either distance or similarity"
            )

        rel_ranks.loc[p] = np.where(pred_metrics.index == p)[0][0]

    rel_ranks = rel_ranks / len(rel_ranks)
    return rel_ranks


def mmd_energy_distance_helper(
    eval,  # an Evaluation objective
    model_name,
    pert_col,
    cov_cols,
    ctrl,
    delim='_',
    kernel='energy_distance',
    gamma=None,
):
    model_adata = eval.adatas[model_name]
    ref_adata = eval.adatas['ref']

    model_adata.obs[pert_col] = model_adata.obs[pert_col].astype('category')
    ref_adata.obs[pert_col] = ref_adata.obs[pert_col].astype('category')

    if len(cov_cols) == 0:
        model_adata.obs['_dummy_cov'] = '1'
        ref_adata.obs['_dummy_cov'] = '1'
        cov_cols = ['_dummy_cov']

    for col in cov_cols:
        assert col in model_adata.obs.columns
        assert col in ref_adata.obs.columns

    if kernel == 'energy_distance':
        kernel_fns = [lambda x, y: - euclidean_distances(x, y)]
    elif kernel == 'rbf_kernel':
        if gamma is None:
            all_gamma = np.logspace(1, -3, num=5)
        elif isinstance(gamma, list):
            all_gamma = np.array(gamma)
        else:
            all_gamma = np.array([gamma])
        kernel_fns = [lambda x, y: rbf_kernel(x, y, gamma=gamma) for gamma in all_gamma]
        print('rbf kernels with gammas:', kernel_fns)
    else:
        raise ValueError('Invalid kernel')

    model_adata_covs = merge_cols(model_adata.obs, cov_cols, delim=delim)
    ref_adata_covs = merge_cols(ref_adata.obs, cov_cols, delim=delim)

    ret = {'cov_pert': [], 'model': [], 'metric': []}
    for cov in model_adata_covs.cat.categories:

        if len(model_adata[model_adata_covs == cov, :].obs[pert_col].unique()) > 1:  # has any perturbations beside control

            model_adata_subset_cov = model_adata[model_adata_covs == cov, :]
            ref_adata_subset_cov = ref_adata[ref_adata_covs == cov, :]
            model_adata_covs_perts = merge_cols(model_adata_subset_cov.obs, [pert_col], delim=delim)
            ref_adata_covs_perts = merge_cols(ref_adata_subset_cov.obs, [pert_col], delim=delim)

            for i, pert in enumerate(model_adata_covs_perts.cat.categories):
                if pert == ctrl:
                    continue

                population_pred = model_adata_subset_cov[model_adata_covs_perts.isin([pert]), :].X
                population_truth = ref_adata_subset_cov[ref_adata_covs_perts.isin([pert]), :].X

                all_mmd = []
                for kernel in kernel_fns:
                    xx = kernel(population_pred, population_pred)
                    xy = kernel(population_pred, population_truth)
                    yy = kernel(population_truth, population_truth)
                    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
                    all_mmd.append(mmd)
                mmd = np.nanmean(all_mmd)

                ret['cov_pert'].append(f'{cov}{delim}{pert}')
                ret['model'].append(model_name)
                ret['metric'].append(mmd)

    eval.mmd_df = pd.DataFrame.from_dict(ret)

    return eval.mmd_df