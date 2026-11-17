#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os as _os
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
_os.environ.setdefault("OMP_NUM_THREADS", "1")

import os
import pickle
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import List, Tuple, Optional, Dict, Any

from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

from contextlib import contextmanager
from threadpoolctl import threadpool_limits


# -------------------------- Utilities --------------------------------------


@contextmanager
def limit_blas_openmp(threads: int):
    t = max(1, int(threads) if threads else 1)
    try:
        with threadpool_limits(limits=t, user_api="blas"):
            with threadpool_limits(limits=t, user_api="openmp"):
                yield
    except Exception:
        with threadpool_limits(limits=t):
            yield


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def beta_tag_str(beta: float) -> str:
    return f"{float(beta):.12g}".replace(".", "_").replace("-", "m")


def build_augmented_design(features: np.ndarray, beta: float) -> np.ndarray:
    Xb64 = np.power(np.asarray(features, dtype=np.float64, order="C"), beta)
    SCg64 = Xb64.mean(axis=1)
    Xb_aug = np.append(Xb64, SCg64.reshape(features.shape[0], 1), axis=1).astype(
        np.float32, copy=False
    )
    return Xb_aug


# ----------------------- Vectorized corr/p with zero-exclusion -------------


try:
    from scipy import stats as _scistats

    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def corr_pvals_exclude_zeros(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    X = np.asarray(X, dtype=np.float64, order="C")
    y = np.asarray(y, dtype=np.float64).ravel()

    n, p = X.shape
    M = X != 0.0
    M_float = M.astype(np.float64)
    n_eff = M_float.sum(axis=0)

    absX = np.abs(X)
    col_max = np.max(absX * M_float, axis=0)
    col_scale = np.where(col_max > 0.0, col_max, 1.0)
    Xs = X / col_scale

    y_scale = np.max(np.abs(y))
    if not np.isfinite(y_scale) or y_scale <= 0.0:
        y_scale = 1.0
    ys = y / y_scale

    sx = (Xs * M_float).sum(axis=0)
    sy = M_float.T @ ys
    sxy = (Xs * (M_float * ys[:, None])).sum(axis=0)
    sx2 = ((Xs * Xs) * M_float).sum(axis=0)
    y2 = ys * ys
    sy2 = M_float.T @ y2

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_x = sx / n_eff
        mean_y = sy / n_eff
        cov_xy = sxy - n_eff * mean_x * mean_y
        var_x = sx2 - n_eff * mean_x * mean_x
        var_y = sy2 - n_eff * mean_y * mean_y
        var_x = np.clip(var_x, 0.0, None)
        var_y = np.clip(var_y, 0.0, None)
        denom = np.sqrt(var_x * var_y)
        r = np.divide(cov_xy, denom, out=np.zeros_like(cov_xy), where=denom > 0)

    df = np.maximum(n_eff - 2.0, 1.0)
    r = np.clip(r, -0.999999, 0.999999)
    t = r * np.sqrt(df / np.maximum(1e-12, (1.0 - r * r)))
    if _HAVE_SCIPY:
        p = 2.0 * _scistats.t.sf(np.abs(t), df)
    else:
        p = np.minimum(1.0, np.maximum(0.0), np.erfc(np.abs(t) / np.sqrt(2.0)))

    r = np.where(n_eff >= 3.0, r, 0.0)
    p = np.where(n_eff >= 3.0, p, 1.0)
    return r.astype(np.float32), p.astype(np.float32), n_eff.astype(np.int32)


# ----------------------- CPM screening -------------------------------------


def cpm_screen_indices(
    features_train: np.ndarray,
    ability_train: np.ndarray,
    p_thresh: float = 1e-3,
    loo_stride: int = 1,
) -> np.ndarray:

    n_train, p = features_train.shape
    keep = np.ones(p, dtype=bool)
    idx_all = np.arange(p, dtype=int)

    loo = KFold(n_splits=n_train)
    for k, (tr, _va) in enumerate(loo.split(features_train, ability_train)):
        if loo_stride > 1 and (k % loo_stride != 0):
            continue
        cols = np.where(keep)[0]
        if cols.size == 0:
            break
        Xtr = features_train[tr][:, cols]
        ytr = ability_train[tr]
        _r, pvals, _n = corr_pvals_exclude_zeros(Xtr, ytr)
        keep[cols] &= pvals <= p_thresh

    return idx_all[keep]


def safe_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if (
        mask.sum() >= 3
        and np.std(y_true[mask]) > 0
        and np.std(y_pred[mask]) > 0
    ):
        r, _ = pearsonr(y_pred[mask], y_true[mask])
        return float(r)
    return float("nan")


# ----------------------- Refit helper on new SC ----------------------------


def refit_cpm_lassocv_on_new_sc(
    X_aug: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    lasso_n_jobs: int = 1,
    loo_stride: int = 5,
    blas_threads: int = 1,
):

    y_tr = y[train_indices]
    y_te = y[test_indices]

    with limit_blas_openmp(blas_threads):
        X_tr = X_aug[train_indices, :]
        X_te = X_aug[test_indices, :]

        kept_idx = cpm_screen_indices(
            X_tr, y_tr, p_thresh=1e-3, loo_stride=loo_stride
        )
        n_selected = int(kept_idx.size)

        if n_selected == 0:
            yhat = np.full_like(y_te, np.nan, dtype=np.float32)
            r = float("nan")
            return yhat, r

        X_tr_sel = X_tr[:, kept_idx]
        X_te_sel = X_te[:, kept_idx]

        lasso = LassoCV(cv=5, max_iter=5000, tol=1e-3, n_jobs=lasso_n_jobs)
        lasso.fit(X_tr_sel, y_tr)
        yhat = lasso.predict(X_te_sel).astype(np.float32)

    r = safe_r(y_te, yhat)
    return yhat, r


# -------------------------- Data I/O ---------------------------------------


def load_data(
    features_csv: str, subject_info_csv: str, outcome: str, return_mask: bool = False
):
    X = pd.read_csv(features_csv, header=None).values
    info = pd.read_csv(subject_info_csv)
    if outcome not in info.columns:
        raise ValueError(f"Outcome '{outcome}' not found in subject_info CSV.")
    y = info[outcome].values.astype(float)

    if "subject_id" in info.columns:
        subject_ids = info["subject_id"].to_numpy()
    else:
        subject_ids = np.arange(X.shape[0], dtype=int)

    if "family_id" in info.columns:
        family_ids = info["family_id"].to_numpy()
    else:
        family_ids = subject_ids.copy()

    mask = np.isfinite(y)
    dropped = int((~mask).sum())
    if dropped > 0:
        print(f"[Load] Dropping {dropped} rows with NaN in outcome '{outcome}'.")
    X = X[mask, :]
    y = y[mask]
    subject_ids = subject_ids[mask]
    family_ids = family_ids[mask]

    if return_mask:
        return X, y, subject_ids, family_ids, mask
    else:
        return X, y, subject_ids, family_ids


# -------------------------- Stage 2 evaluation -----------------------------


def evaluate_saved_models_on_given_sc(
    features_drop: np.ndarray,
    features_rewire: np.ndarray,
    ability: np.ndarray,
    out_dir: str,
    outcome_name: str,
):

    base = os.path.join(out_dir, "legacy_cpm")
    models_pkl = os.path.join(base, "models", "all_models_meta.pkl")
    if not os.path.exists(models_pkl):
        raise FileNotFoundError(f"找不到已训练模型信息: {models_pkl}")

    with open(models_pkl, "rb") as f:
        all_models_meta: List[Dict[str, Any]] = pickle.load(f)

    models_meta = [m for m in all_models_meta if m.get("variant") == "beta_star"]
    if not models_meta:
        raise RuntimeError("在 all_models_meta.pkl 中没有找到 variant == 'beta_star' 的模型.")

    unique_betas = sorted({float(m["beta"]) for m in models_meta})

    design_drop: Dict[float, np.ndarray] = {}
    design_rewire: Dict[float, np.ndarray] = {}

    for b in unique_betas:
        design_drop[b] = build_augmented_design(features_drop, b)
        design_rewire[b] = build_augmented_design(features_rewire, b)

    lasso_n_jobs = 18
    loo_stride = 5
    blas_threads = 1

    results_dir = os.path.join(base, "test_eval_stage2_given_sc")
    preds_dir = os.path.join(base, "predictions_stage2_given_sc")
    ensure_dir(results_dir)
    ensure_dir(preds_dir)

    rows = []

    y_true_all_drop: List[float] = []
    y_pred_all_drop: List[float] = []
    y_true_all_rewire: List[float] = []
    y_pred_all_rewire: List[float] = []

    for m in models_meta:
        outer_id = m["outer_id"]
        rep = int(m["repeat"])
        fold = int(m["fold"])
        beta_val = float(m["beta"])
        beta_design_drop = design_drop[beta_val]
        beta_design_rewire = design_rewire[beta_val]

        train_indices = np.asarray(m["train_indices"], dtype=np.int64)
        test_indices = np.asarray(m["test_indices"], dtype=np.int64)
        y_test = ability[test_indices].astype(np.float32)

        yhat_drop, r_drop = refit_cpm_lassocv_on_new_sc(
            beta_design_drop,
            ability,
            train_indices,
            test_indices,
            lasso_n_jobs=lasso_n_jobs,
            loo_stride=loo_stride,
            blas_threads=blas_threads,
        )

        yhat_rewire, r_rewire = refit_cpm_lassocv_on_new_sc(
            beta_design_rewire,
            ability,
            train_indices,
            test_indices,
            lasso_n_jobs=lasso_n_jobs,
            loo_stride=loo_stride,
            blas_threads=blas_threads,
        )

        rows.append(
            {
                "outer_id": outer_id,
                "repeat": rep,
                "fold": fold,
                "beta": beta_val,
                "r_drop": r_drop,
                "r_rewire": r_rewire,
            }
        )

        y_true_all_drop.extend(y_test.tolist())
        y_pred_all_drop.extend(yhat_drop.tolist())
        y_true_all_rewire.extend(y_test.tolist())
        y_pred_all_rewire.extend(yhat_rewire.tolist())

        np.savez_compressed(
            os.path.join(preds_dir, f"pred_stage2_{outer_id}.npz"),
            outer_id=outer_id,
            repeat=rep,
            fold=fold,
            beta=beta_val,
            test_indices=test_indices,
            y_true=y_test,
            y_pred_drop=yhat_drop,
            y_pred_rewire=yhat_rewire,
        )

    df_rows = pd.DataFrame(rows)
    df_rows.sort_values(["repeat", "fold"], inplace=True)
    df_rows.to_csv(
        os.path.join(results_dir, f"per_outer_r_{outcome_name}.csv"),
        index=False,
    )

    def agg_r(y_true_list, y_pred_list) -> float:
        if not y_true_list or not y_pred_list:
            return float("nan")
        ya = np.asarray(y_true_list, dtype=np.float64)
        yp = np.asarray(y_pred_list, dtype=np.float64)
        return safe_r(ya, yp)

    summary = {
        "outcome": outcome_name,
        "r_drop_all": agg_r(y_true_all_drop, y_pred_all_drop),
        "r_rewire_all": agg_r(y_true_all_rewire, y_pred_all_rewire),
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(results_dir, f"overall_r_{outcome_name}.csv"),
        index=False,
    )

    print(f"[Stage2] Evaluation on SC_drop / SC_rewire saved under: {results_dir}")
    return {"base_dir": base, "results_dir": results_dir}


# -------------------------- Main -------------------------------------------

def main():
    features_drop_csv = "./Data/SC/SC_commit2_weak_rewire_ICC_re_02.csv"
    features_rewire_csv = "./Data/SC/SC_commit2_weak_rewire_ICC_unre_02.csv"
    subject_info_csv = "./Data/HCP_abilities/HCPabilities.csv"
    outcomes = ["g","mem","cry"]

    output_dir = "./outputs"

    for outcome in outcomes:
        print(f"==== Stage2: outcome={outcome} ====")

        X_drop, y, subject_ids, family_ids, mask = load_data(
            features_drop_csv, subject_info_csv, outcome, return_mask=True
        )

        X_rewire_full = pd.read_csv(features_rewire_csv, header=None).values
        if X_rewire_full.shape[0] != mask.shape[0]:
            raise ValueError("SC_rewire.csv 行数与 SC_drop.csv / HCP2 不一致.")
        X_rewire = X_rewire_full[mask, :]
        if X_rewire.shape[1] != X_drop.shape[1]:
            raise ValueError("SC_rewire.csv 特征列数与 SC_drop.csv 不一致。")

        out_dir = f"{output_dir}/{outcome}"

        _ = evaluate_saved_models_on_given_sc(
            features_drop=X_drop,
            features_rewire=X_rewire,
            ability=y,
            out_dir=out_dir,
            outcome_name=outcome,
        )


if __name__ == "__main__":
    main()
