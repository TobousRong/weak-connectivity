#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os as _os
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
_os.environ.setdefault("OMP_NUM_THREADS", "1")

import os
import time
import pickle
import warnings
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from contextlib import contextmanager
import gc


# -------------------------- Utilities --------------------------------------

@contextmanager
def limit_blas_openmp(threads: int):
    t = max(1, int(threads) if threads else 1)
    try:
        with threadpool_limits(limits=t, user_api='blas'):
            with threadpool_limits(limits=t, user_api='openmp'):
                yield
    except Exception:
        with threadpool_limits(limits=t):
            yield


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_pickle(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def resolve_n_jobs(n_jobs_outer: int) -> int:
    if n_jobs_outer is None or int(n_jobs_outer) == 0:
        return 1
    if int(n_jobs_outer) < 0:
        cpu = os.cpu_count() or 1
        return max(1, cpu)
    return max(1, int(n_jobs_outer))


def beta_tag_str(beta: float) -> str:
    # 用于文件名：0.1 -> 0_1, -0.5 -> m0_5
    return f"{float(beta):.12g}".replace(".", "_").replace("-", "m")


# -------------------------- Grouped Outer Splits ---------------------------

def make_outer_splits_repeated_group_kfold(
    groups: np.ndarray,
    n_splits: int = 10,
    n_repeats: int = 10,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray, str, int, int]]:
    rng = np.random.RandomState(random_state)
    groups = np.asarray(groups)
    unique, counts = np.unique(groups, return_counts=True)

    splits: List[Tuple[np.ndarray, np.ndarray, str, int, int]] = []

    for rep in range(n_repeats):
        order = rng.permutation(len(unique))
        ug = unique[order]
        uc = counts[order]

        fold_groups = [[] for _ in range(n_splits)]
        fold_sizes = [0 for _ in range(n_splits)]
        for g, c in zip(ug, uc):
            k = int(np.argmin(fold_sizes))
            fold_groups[k].append(g)
            fold_sizes[k] += int(c)

        for fold in range(n_splits):
            test_mask = np.isin(groups, fold_groups[fold])
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]
            outer_id = f"repeat{rep}_fold{fold}"
            splits.append((train_idx, test_idx, outer_id, rep, fold))

    return splits


# ----------------------- Vectorized corr/p with zero-exclusion -------------

try:
    from scipy import stats as _scistats
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def corr_pvals_exclude_zeros(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64, order='C')
    y = np.asarray(y, dtype=np.float64).ravel()

    n, p = X.shape
    M = (X != 0.0)
    M_float = M.astype(np.float64)
    n_eff = M_float.sum(axis=0)

    # absX = np.abs(X)
    # col_max = np.max(absX * M_float, axis=0)
    # col_scale = np.where(col_max > 0.0, col_max, 1.0)
    # Xs = X / col_scale
    #
    # y_scale = np.max(np.abs(y))
    # if not np.isfinite(y_scale) or y_scale <= 0.0:
    #     y_scale = 1.0
    # ys = y / y_scale
    #
    # sx = (Xs * M_float).sum(axis=0)
    # sy = M_float.T @ ys
    # sxy = (Xs * (M_float * ys[:, None])).sum(axis=0)
    # sx2 = ((Xs * Xs) * M_float).sum(axis=0)
    # y2 = ys * ys
    # sy2 = M_float.T @ y2
    #
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     mean_x = sx / n_eff
    #     mean_y = sy / n_eff
    #     cov_xy = sxy - n_eff * mean_x * mean_y
    #     var_x = sx2 - n_eff * mean_x * mean_x
    #     var_y = sy2 - n_eff * mean_y * mean_y
    #     var_x = np.clip(var_x, 0.0, None)
    #     var_y = np.clip(var_y, 0.0, None)
    #     denom = np.sqrt(var_x * var_y)
    #     r = np.divide(cov_xy, denom, out=np.zeros_like(cov_xy), where=denom > 0)

    ## pearson
    # df = np.maximum(n_eff - 2.0, 1.0)
    # r = np.clip(r, -0.999999, 0.999999)
    # t = r * np.sqrt(df / np.maximum(1e-12, (1.0 - r * r)))
    # if _HAVE_SCIPY:
    #     p = 2.0 * _scistats.t.sf(np.abs(t), df)
    # else:
    #     p = np.minimum(1.0, np.maximum(0.0), np.erfc(np.abs(t) / np.sqrt(2.0)))

    ## spearmanr
    r = np.zeros(X.shape[1], dtype=np.float64)
    p = np.ones(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        mask = M[:, j] & np.isfinite(X[:, j]) & np.isfinite(y)
        n_eff[j] = mask.sum()

        if n_eff[j] >= 3:
            result = _scistats.spearmanr(X[mask, j], y[mask])
            if np.isfinite(result.statistic) and np.isfinite(result.pvalue):
                r[j] = result.statistic
                p[j] = result.pvalue

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
        keep[cols] &= (pvals <= p_thresh)

    return idx_all[keep]


def cpm_screen_once(
    features_train: np.ndarray,
    ability_train: np.ndarray,
    p_thresh: float = 1e-3,
) -> np.ndarray:
    _r, pvals, _n = corr_pvals_exclude_zeros(features_train, ability_train)
    return np.where(pvals <= p_thresh)[0]


# ----------------------- Design transforms / memmap ------------------------

def persist_designs_memmap(
    features_initial: np.ndarray,
    betas: List[float],
    mmap_dir: str,
) -> Dict[float, str]:
    ensure_dir(mmap_dir)
    N, P0 = features_initial.shape
    file_map: Dict[float, str] = {}

    for beta in betas:
        beta_tag = beta_tag_str(beta)
        out_path = os.path.join(mmap_dir, f"X_aug_beta_{beta_tag}.npy")
        if not os.path.exists(out_path):
            Xb64 = np.power(np.asarray(features_initial, dtype=np.float64, order='C'), beta)
            SCg64 = Xb64.mean(axis=1)
            Xb_aug = np.append(Xb64, SCg64.reshape(N, 1), axis=1).astype(np.float32, copy=False)
            np.save(out_path, Xb_aug)
            del Xb64, SCg64, Xb_aug
        file_map[beta] = out_path

    return file_map


@contextmanager
def open_memmap_ondemand(path: str):
    arr = np.load(path, mmap_mode='r')
    try:
        yield arr
    finally:
        try:
            if hasattr(arr, '_mmap') and arr._mmap is not None:
                arr._mmap.close()
        except Exception:
            pass
        del arr
        gc.collect()


# ----------------------- Worker (fit + predict + collect) ------------------

def _fit_eval_on_outer(
    X_aug: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    train_outer: np.ndarray,
    test_outer: np.ndarray,
    beta_value: float,
    outer_id: str,
    lasso_n_jobs: int,
    loo_stride: int,
    inner_screen_mode: str,
    blas_threads: int,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    with limit_blas_openmp(blas_threads):
        X_tr = X_aug[train_outer, :]
        X_te = X_aug[test_outer, :]

        kept_idx = cpm_screen_indices(X_tr, y_tr, p_thresh=1e-3, loo_stride=loo_stride)
        n_selected = int(kept_idx.size)
        n_total = int(X_tr.shape[1])
        print(f"[Screen] Outer {outer_id}, beta {beta_value}: selected {n_selected}/{n_total}")

        if n_selected == 0:
            return np.full_like(y_te, np.nan, dtype=np.float32), np.nan, {
                "n_selected": 0, "n_total": n_total,
                "kept_idx": np.array([], dtype=np.int32),
                "coef": np.array([], dtype=np.float32),
                "coef_full": np.zeros(n_total, dtype=np.float32),
                "intercept": 0.0
            }

        X_tr_sel = X_tr[:, kept_idx]
        X_te_sel = X_te[:, kept_idx]

        lasso = LassoCV(cv=5, max_iter=5000, tol=1e-3, n_jobs=lasso_n_jobs)
        lasso.fit(X_tr_sel, y_tr)
        yhat = lasso.predict(X_te_sel).astype(np.float32)

        coef = lasso.coef_.astype(np.float32, copy=False)
        intercept = float(lasso.intercept_)
        coef_full = np.zeros(n_total, dtype=np.float32)
        coef_full[kept_idx] = coef

        if np.std(yhat) > 0 and np.std(y_te) > 0:
            r, _ = pearsonr(yhat, y_te)
        else:
            r = np.nan

        return yhat, float(r), {
            "n_selected": n_selected, "n_total": n_total,
            "kept_idx": kept_idx.astype(np.int32, copy=False),
            "coef": coef, "coef_full": coef_full, "intercept": intercept
        }


def process_one_outer_split(task: Dict[str, Any]) -> Dict[str, Any]:
    betas: List[float] = task["betas"]
    mmap_map: Dict[float, str] = task["mmap_map"]
    y: np.ndarray = task["y"]
    train_outer: np.ndarray = task["train_outer"]
    test_outer: np.ndarray = task["test_outer"]
    outer_id: str = task["outer_id"]
    rep: int = task["rep"]
    fold: int = task["fold"]
    subject_ids: np.ndarray = task["subject_ids"]
    dirs: Dict[str, str] = task["dirs"]
    n_splits_inner: int = task["n_splits_inner"]
    seed: int = task["seed"]
    lasso_n_jobs: int = task["lasso_n_jobs"]
    loo_stride: int = task["loo_stride"]
    inner_screen_mode: str = task["inner_screen_mode"]
    eval_all_betas_on_test: int = task["eval_all_betas_on_test"]
    parallel_mode: str = task["parallel_mode"]
    n_jobs_inner_beta: int = task["n_jobs_inner_beta"]
    blas_threads: int = task["blas_threads"]

    y_tr = y[train_outer]
    y_te = y[test_outer]

    def save_retained(model_key: str, kept_idx: np.ndarray, meta: Dict[str, Any]) -> Optional[str]:
        try:
            ids_path = os.path.join(dirs["ids"], f"{outer_id}__{model_key}.npy")
            np.save(ids_path, kept_idx.astype(np.int32, copy=False))

            w_path = os.path.join(dirs["weights"], f"{outer_id}__{model_key}.npz")
            np.savez_compressed(
                w_path,
                kept_idx=kept_idx.astype(np.int32, copy=False),
                coef=np.asarray(meta.get("coef", np.array([], dtype=np.float32)), dtype=np.float32),
                intercept=np.float32(meta.get("intercept", 0.0)),
                n_total=np.int32(meta.get("n_total", 0)),
                coef_full=np.asarray(meta.get("coef_full", np.array([], dtype=np.float32)), dtype=np.float32),
            )
            return w_path
        except Exception as e:
            print(f"[Warn] Save retained failed for {model_key} on {outer_id}: {e}")
            return None

    inner_kf = KFold(n_splits=n_splits_inner, shuffle=True,
                     random_state=seed + rep * 1000 + fold)
    inner_splits_rel = list(inner_kf.split(train_outer))

    inner_payload = []
    for i, (tr_rel, va_rel) in enumerate(inner_splits_rel):
        tr_abs = train_outer[tr_rel]
        va_abs = train_outer[va_rel]
        inner_payload.append({
            "fold": i,
            "train_indices": tr_abs.tolist(),
            "val_indices": va_abs.tolist(),
            "train_subject_ids": subject_ids[tr_abs].tolist(),
            "val_subject_ids": subject_ids[va_abs].tolist(),
        })
    save_pickle(inner_payload, os.path.join(dirs["splits"], f"inner_splits_{outer_id}.pkl"))

    def _inner_score_for_beta(beta: float) -> Dict[str, Any]:
        with open_memmap_ondemand(mmap_map[beta]) as Xb_aug:
            val_scores = []
            for (tr_rel, va_rel) in inner_splits_rel:
                tr_abs = train_outer[tr_rel]
                va_abs = train_outer[va_rel]
                X_tr_in = Xb_aug[tr_abs, :]
                y_tr_in = y[tr_abs]
                X_va_in = Xb_aug[va_abs, :]
                y_va_in = y[va_abs]

                if inner_screen_mode.lower() == "once":
                    kept_idx = cpm_screen_once(X_tr_in, y_tr_in, p_thresh=1e-3)
                else:
                    kept_idx = cpm_screen_indices(X_tr_in, y_tr_in, p_thresh=1e-3, loo_stride=task["loo_stride"])

                if kept_idx.size == 0:
                    val_scores.append(np.nan)
                    continue

                X_tr_sel = X_tr_in[:, kept_idx]
                X_va_sel = X_va_in[:, kept_idx]

                with limit_blas_openmp(blas_threads):
                    lasso = LassoCV(cv=5, max_iter=5000, tol=1e-3, n_jobs=lasso_n_jobs)
                    try:
                        lasso.fit(X_tr_sel, y_tr_in)
                        yhat_va = lasso.predict(X_va_sel)
                        if np.std(yhat_va) > 0 and np.std(y_va_in) > 0:
                            r, _ = pearsonr(yhat_va, y_va_in)
                        else:
                            r = np.nan
                    except Exception:
                        r = np.nan
                val_scores.append(r)

            val_scores = np.asarray(val_scores, dtype=np.float32)
            return {
                "beta": beta,
                "mean_validation_r": float(np.nanmean(val_scores)) if val_scores.size else np.nan,
                "std_validation_r": float(np.nanstd(val_scores, ddof=1)) if np.sum(np.isfinite(val_scores)) > 1 else np.nan,
                "n_folds": int(np.sum(np.isfinite(val_scores)))
            }

    if parallel_mode == "beta" and n_jobs_inner_beta > 1:
        inner_rows = Parallel(n_jobs=n_jobs_inner_beta, backend="loky")(
            delayed(_inner_score_for_beta)(beta) for beta in betas
        )
    else:
        inner_rows = [_inner_score_for_beta(beta) for beta in betas]

    df_inner = pd.DataFrame(inner_rows).sort_values("beta").reset_index(drop=True)
    if df_inner["n_folds"].max() == 0 or not np.any(np.isfinite(df_inner["mean_validation_r"])):
        beta_star = float(betas[int(np.argmin(np.abs(np.asarray(betas) - 1.0)))])
    else:
        beta_star = float(df_inner.loc[df_inner["mean_validation_r"].idxmax(), "beta"])

    df_inner["selected"] = df_inner["beta"] == beta_star
    df_inner["outer_id"] = outer_id
    ensure_dir(dirs["tuning"])
    df_inner.to_csv(os.path.join(dirs["tuning"], f"{outer_id}.csv"), index=False)

    all_test_rows_long: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    concat_true_pred: Dict[float, Tuple[List[float], List[float]]] = {b: ([], []) for b in betas}
    legacy_predictions: List[Tuple[float, np.ndarray, np.ndarray]] = []
    saved_weight_records: List[Tuple[str, str]] = []
    models_meta: List[Dict[str, Any]] = []

    def _eval_one_beta(beta: float, do_save: bool = True) -> Tuple[float, float, int, int, np.ndarray]:
        with open_memmap_ondemand(mmap_map[beta]) as Xb_aug:
            yhat, r, meta = _fit_eval_on_outer(
                X_aug=Xb_aug,
                y_tr=y_tr, y_te=y_te,
                train_outer=train_outer, test_outer=test_outer,
                beta_value=beta, outer_id=outer_id,
                lasso_n_jobs=lasso_n_jobs,
                loo_stride=task["loo_stride"],
                inner_screen_mode=inner_screen_mode,
                blas_threads=blas_threads,
            )
            n_sel = meta.get("n_selected", 0)
            n_tot = meta.get("n_total", Xb_aug.shape[1])

            if do_save and n_sel > 0:
                mk = f"beta{beta_tag_str(beta)}"
                w_path = save_retained(mk, meta["kept_idx"], meta)
                if w_path is not None:
                    saved_weight_records.append((mk, w_path))

            return r, beta, n_sel, n_tot, yhat

    if int(eval_all_betas_on_test) == 1:
        if parallel_mode == "beta" and n_jobs_inner_beta > 1:
            eval_results = Parallel(n_jobs=n_jobs_inner_beta, backend="loky")(
                delayed(_eval_one_beta)(beta, True) for beta in betas
            )
        else:
            eval_results = [_eval_one_beta(beta, True) for beta in betas]

        for (r_b, beta, n_sel, n_tot, yhat_b) in eval_results:
            all_test_rows_long.append({
                "outer_id": outer_id, "repeat": rep, "fold": fold,
                "beta": beta, "r": float(r_b), "n_selected": int(n_sel)
            })
            if yhat_b is not None and np.any(np.isfinite(yhat_b)):
                yt, yp = concat_true_pred[beta]
                yt.extend(y_te.tolist())
                yp.extend(yhat_b.tolist())
                legacy_predictions.append((beta, test_outer.copy(), yhat_b.copy()))
            selected_rows.append({
                "outer_id": outer_id, "beta": beta, "n_selected": int(n_sel), "n_total": int(n_tot)
            })
    else:
        _eval_one_beta(beta_star, True)
        if 1.0 in betas:
            _eval_one_beta(1.0, True)

    def _eval_only(beta: float) -> float:
        with open_memmap_ondemand(mmap_map[beta]) as Xb_aug:
            yhat, r, meta = _fit_eval_on_outer(
                X_aug=Xb_aug, y_tr=y_tr, y_te=y_te,
                train_outer=train_outer, test_outer=test_outer,
                beta_value=beta, outer_id=outer_id,
                lasso_n_jobs=lasso_n_jobs,
                loo_stride=task["loo_stride"],
                inner_screen_mode=inner_screen_mode,
                blas_threads=blas_threads,
            )
            return float(r)

    r_star = _eval_only(beta_star)
    r_b1 = _eval_only(1.0) if 1.0 in betas else np.nan

    def _has_model(model_key: str) -> bool:
        return any(mk == model_key for mk, _ in saved_weight_records)

    mk_star = f"beta{beta_tag_str(beta_star)}"
    if _has_model(mk_star):
        models_meta.append({
            "outer_id": outer_id,
            "repeat": rep,
            "fold": fold,
            "variant": "beta_star",
            "beta": beta_star,
            "r": r_star,
            "model_key": mk_star,
            "weights_path": os.path.join(dirs["weights"], f"{outer_id}__{mk_star}.npz"),
            "ids_path": os.path.join(dirs["ids"], f"{outer_id}__{mk_star}.npy"),
            "train_indices": train_outer.tolist(),
            "test_indices": test_outer.tolist(),
            "train_subject_ids": subject_ids[train_outer].tolist(),
            "test_subject_ids": subject_ids[test_outer].tolist(),
        })

    ensure_dir(dirs["testeval"])
    pd.DataFrame([
        {"outer_id": outer_id, "beta": "beta_star", "beta_value": beta_star, "r": r_star},
        {"outer_id": outer_id, "beta": "beta_1",    "beta_value": 1.0,       "r": r_b1},
    ]).to_csv(os.path.join(dirs["testeval"], f"{outer_id}.csv"), index=False)

    return {
        "outer_id": outer_id, "rep": rep, "fold": fold,
        "inner_table": df_inner,
        "long_rows": all_test_rows_long,
        "selected_rows": selected_rows,
        "concat_true_pred": concat_true_pred,
        "beta_star_vs_beta1": {
            "outer_id": outer_id, "repeat": rep, "fold": fold,
            "beta_star_value": beta_star, "r_beta_star": r_star, "r_beta1": r_b1,
            "r_diff": (r_star - r_b1) if (np.isfinite(r_star) and np.isfinite(r_b1)) else np.nan,
        },
        "legacy_predictions": legacy_predictions,
        "saved_weight_records": saved_weight_records,
        "models_meta": models_meta,
    }


# ----------------------------- Core driver (training) ----------------------


def prediction_lassocv_grouped(
    features_initial: np.ndarray,
    ability: np.ndarray,
    betas: List[float],
    subject_ids: np.ndarray,
    family_ids: np.ndarray,
    out_dir: str,
    outcome_name: str,
    n_splits_outer: int = 10,
    n_repeats_outer: int = 10,
    n_splits_inner: int = 10,
    seed: int = 42,
    parallel_mode: str = "outer",
    n_jobs_outer: int = -1,
    blas_threads_per_worker: int = 1,
    lasso_n_jobs: Optional[int] = None,
    use_memmap: int = 1,
    mmap_cache_dir: Optional[str] = None,
    keep_mmap_cache: int = 1,
    compute_legacy_allcorr: int = 0,
    loo_stride: int = 5,
    inner_screen_mode: str = "once",
    eval_all_betas_on_test: int = 1,
    consolidate_outputs: int = 0,
):
    t0 = time.time()

    base = os.path.join(out_dir, "legacy_cpm")
    dirs = {
        "retain_avg": os.path.join(base, "retained_avg_weights"),
        "weights": os.path.join(base, "retained_weights"),
        "ids": os.path.join(base, "retained_ids"),
        "tables": os.path.join(base, "tables"),
        "splits": os.path.join(base, "splits"),
        "tuning": os.path.join(base, "tuning", "inner_cv_selection"),
        "testeval": os.path.join(base, "test_eval_main"),
        "consolidated": os.path.join(base, "consolidated"),
        "preds": os.path.join(base, "predictions"),
        "mmap_cache": os.path.join(base, "mmap_cache") if mmap_cache_dir is None else mmap_cache_dir,
        "models": os.path.join(base, "models"),
    }
    for d in dirs.values():
        ensure_dir(d)

    betas = [float(b) for b in betas]
    seen = set()
    betas = [b for b in betas if (b not in seen and not seen.add(b))]

    outer_splits = make_outer_splits_repeated_group_kfold(
        groups=family_ids, n_splits=n_splits_outer, n_repeats=n_repeats_outer, random_state=seed
    )
    outer_payload = []
    for train_idx, test_idx, outer_id, rep, fold in outer_splits:
        outer_payload.append({
            "outer_id": outer_id, "repeat": rep, "fold": fold,
            "train_indices": train_idx.tolist(), "test_indices": test_idx.tolist(),
            "train_subject_ids": subject_ids[train_idx].tolist(),
            "test_subject_ids": subject_ids[test_idx].tolist()
        })
    save_pickle(outer_payload, os.path.join(dirs["splits"], "outer_splits.pkl"))

    if int(use_memmap) == 1:
        mmap_map = persist_designs_memmap(features_initial, betas, dirs["mmap_cache"])
    else:
        mmap_map = {}
        for beta in betas:
            Xb64 = np.power(np.asarray(features_initial, dtype=np.float64, order='C'), beta)
            SCg64 = Xb64.mean(axis=1)
            Xb_aug = np.append(Xb64, SCg64.reshape(features_initial.shape[0], 1), axis=1).astype(np.float32, copy=False)
            beta_tag = beta_tag_str(beta)
            tmp_path = os.path.join(dirs["mmap_cache"], f"X_aug_beta_{beta_tag}.npy")
            ensure_dir(os.path.dirname(tmp_path))
            np.save(tmp_path, Xb_aug)
            mmap_map[beta] = tmp_path
            del Xb64, SCg64, Xb_aug

    outer_workers = resolve_n_jobs(n_jobs_outer)
    blas_threads = max(1, int(blas_threads_per_worker))
    if lasso_n_jobs is None:
        lasso_n_jobs_eff = 1 if (parallel_mode in ["outer", "beta"] and outer_workers > 1) else -1
    else:
        lasso_n_jobs_eff = int(lasso_n_jobs)

    tasks = []
    for (train_outer, test_outer, outer_id, rep, fold) in outer_splits:
        tasks.append({
            "betas": betas,
            "mmap_map": mmap_map,
            "y": ability,
            "train_outer": train_outer,
            "test_outer": test_outer,
            "outer_id": outer_id,
            "rep": rep,
            "fold": fold,
            "subject_ids": subject_ids,
            "dirs": dirs,
            "n_splits_inner": n_splits_inner,
            "seed": seed,
            "lasso_n_jobs": lasso_n_jobs_eff,
            "loo_stride": loo_stride,
            "inner_screen_mode": inner_screen_mode,
            "eval_all_betas_on_test": eval_all_betas_on_test,
            "parallel_mode": parallel_mode,
            "n_jobs_inner_beta": outer_workers if parallel_mode == "beta" else 1,
            "blas_threads": blas_threads,
        })

    if parallel_mode == "outer" and outer_workers > 1:
        results = Parallel(n_jobs=outer_workers, backend="loky")(
            delayed(process_one_outer_split)(task) for task in tasks
        )
    else:
        results = [process_one_outer_split(task) for task in tasks]

    if int(consolidate_outputs) == 1:
        ensure_dir(dirs["consolidated"])
        ensure_dir(dirs["tables"])
        ensure_dir(dirs["preds"])

    all_test_rows_long: List[Dict[str, Any]] = []
    beta_star_vs_beta1_rows: List[Dict[str, Any]] = []
    selected_rows_all: List[Dict[str, Any]] = []
    concat_y_true_by_beta: Dict[float, List[float]] = {b: [] for b in betas}
    concat_y_pred_by_beta: Dict[float, List[float]] = {b: [] for b in betas}
    predictions_by_beta: Dict[float, np.ndarray] = {}
    saved_weight_files_by_model: Dict[str, List[str]] = {}
    all_models_meta: List[Dict[str, Any]] = []

    if int(compute_legacy_allcorr) == 1:
        predictions_by_beta = {beta: np.full(ability.size, np.nan, dtype=np.float32) for beta in betas}

    for res in results:
        all_test_rows_long.extend(res["long_rows"])
        beta_star_vs_beta1_rows.append(res["beta_star_vs_beta1"])
        selected_rows_all.extend(res["selected_rows"])

        for beta, (yt, yp) in res["concat_true_pred"].items():
            concat_y_true_by_beta[beta].extend(yt)
            concat_y_pred_by_beta[beta].extend(yp)

        if int(compute_legacy_allcorr) == 1:
            for (beta, test_idx, yhat) in res["legacy_predictions"]:
                predictions_by_beta[beta][test_idx] = yhat

        for mk, path in res.get("saved_weight_records", []):
            saved_weight_files_by_model.setdefault(mk, []).append(path)

        all_models_meta.extend(res.get("models_meta", []))

    if int(consolidate_outputs) == 1 and len(all_test_rows_long) > 0:
        df_long = pd.DataFrame(all_test_rows_long)
        df_long.sort_values(["outer_id", "beta"], inplace=True)
        df_long.to_csv(os.path.join(dirs["consolidated"], "outer_test_all_betas_long.csv"), index=False)
        try:
            df_wide = df_long.pivot(index="outer_id", columns="beta", values="r").sort_index(axis=1)
            df_wide.to_csv(os.path.join(dirs["consolidated"], "outer_test_all_betas_wide.csv"))
        except Exception:
            warnings.warn("Pivot to wide format failed; wide CSV not written.")

    if int(consolidate_outputs) == 1 and len(beta_star_vs_beta1_rows) > 0:
        df_comp = pd.DataFrame(beta_star_vs_beta1_rows)
        df_comp.sort_values(["repeat", "fold"], inplace=True)
        df_comp.to_csv(os.path.join(dirs["consolidated"], "beta_star_vs_beta1.csv"), index=False)

    if int(consolidate_outputs) == 1:
        header = ",".join([str(b) for b in betas])
        all_corr = []
        all_p = []
        for beta in betas:
            y_true_all = np.asarray(concat_y_true_by_beta[beta], dtype=float)
            y_pred_all = np.asarray(concat_y_pred_by_beta[beta], dtype=float)
            mask = np.isfinite(y_true_all) & np.isfinite(y_pred_all)
            if np.sum(mask) >= 3 and np.std(y_true_all[mask]) > 0 and np.std(y_pred_all[mask]) > 0:
                r, p = pearsonr(y_pred_all[mask], y_true_all[mask])
            else:
                r, p = (np.nan, np.nan)
            all_corr.append(r)
            all_p.append(p)
        np.savetxt(os.path.join(dirs["consolidated"], "corr_vs_beta_one_row.csv"),
                   np.asarray([all_corr], dtype=float), delimiter=',', header=header, comments='')
        np.savetxt(os.path.join(dirs["consolidated"], "pval_vs_beta_one_row.csv"),
                   np.asarray([all_p], dtype=float), delimiter=',', header=header, comments='')

        if int(compute_legacy_allcorr) == 1:
            legacy_all_corr = []
            legacy_all_p = []
            for beta in betas:
                yhat_full = predictions_by_beta[beta]
                mask = np.isfinite(yhat_full)
                if np.sum(mask) >= 3 and np.std(ability[mask]) > 0 and np.std(yhat_full[mask]) > 0:
                    r, p = pearsonr(yhat_full[mask], ability[mask])
                else:
                    r, p = (np.nan, np.nan)
                legacy_all_corr.append(r)
                legacy_all_p.append(p)
            np.savetxt(os.path.join(dirs["tables"], f"{outcome_name}_all_corr.csv"),
                       np.asarray([legacy_all_corr], dtype=float), delimiter=',', header=header, comments='')
            np.savetxt(os.path.join(dirs["tables"], f"{outcome_name}_all_pval.csv"),
                       np.asarray([legacy_all_p], dtype=float), delimiter=',', header=header, comments='')
        else:
            np.savetxt(os.path.join(dirs["tables"], f"{outcome_name}_all_corr.csv"),
                       np.full((1, len(betas)), np.nan), delimiter=',', header=header, comments='')
            np.savetxt(os.path.join(dirs["tables"], f"{outcome_name}_all_pval.csv"),
                       np.full((1, len(betas)), np.nan), delimiter=',', header=header, comments='')

        if len(selected_rows_all) > 0:
            pd.DataFrame(selected_rows_all).to_csv(
                os.path.join(dirs["tables"], f"{outcome_name}_selected_features.csv"), index=False
            )

    for mk, paths in saved_weight_files_by_model.items():
        coef_sum = None
        count = 0
        for pth in paths:
            try:
                data = np.load(pth)
                v = np.asarray(data["coef_full"], dtype=np.float32)
                if coef_sum is None:
                    coef_sum = np.zeros_like(v, dtype=np.float64)
                coef_sum += v.astype(np.float64, copy=False)
                count += 1
            except Exception as e:
                print(f"[Warn] Reading weights failed ({pth}): {e}")
        if coef_sum is not None and count > 0:
            avg = (coef_sum / float(count)).astype(np.float32)
            np.save(os.path.join(dirs["retain_avg"], f"avg__{mk}.npy"), avg)

    if all_models_meta:
        save_pickle(all_models_meta, os.path.join(dirs["models"], "all_models_meta.pkl"))
        light_rows = []
        for m in all_models_meta:
            row = {k: v for k, v in m.items()
                   if k not in ("train_indices", "test_indices", "train_subject_ids", "test_subject_ids")}
            light_rows.append(row)
        pd.DataFrame(light_rows).to_csv(os.path.join(dirs["models"], "all_models_meta.csv"), index=False)

    if int(use_memmap) == 1 and int(keep_mmap_cache) == 0:
        for beta, path in mmap_map.items():
            try:
                os.remove(path)
            except Exception:
                pass

    print(f"[Save] Results saved under: {base}  (elapsed {time.time()-t0:.1f}s)")
    return {"base_dir": base}


# ----------------------------- Data I/O ------------------------------------

def load_data(features_csv: str, subject_info_csv: str, outcome: str, return_mask: bool = False):
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


# ----------------------------- Main ----------------------------------------

def main():
    features_csv = "./Data/SC/SC_10M.csv"
    subject_info_csv = "./Data/HCP_abilities/HCPabilities.csv"
    outcomes = ["g","mem","cry"]

    output_dir = "./outputs"

    n_splits_outer = 10
    n_repeats_outer = 10
    n_splits_inner = 10
    seed = 42

    betas =[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    lasso_n_jobs = 10
    compute_legacy_allcorr = 1
    loo_stride = 5
    inner_screen_mode = "once"

    parallel_mode = "outer"
    n_jobs_outer = 15
    blas_threads_per_worker = 1
    use_memmap = 1
    keep_mmap_cache = 1
    eval_all_betas_on_test = 1
    consolidate_outputs = 1

    for outcome in outcomes:
        print(f"==== 开始处理 outcome: {outcome} ====")

        X, y, subject_ids, family_ids, mask = load_data(features_csv, subject_info_csv, outcome, return_mask=True)
        out_dir = f"{output_dir}/{outcome}"

        _ = prediction_lassocv_grouped(
            features_initial=X,
            ability=y,
            betas=betas,
            subject_ids=subject_ids,
            family_ids=family_ids,
            out_dir=out_dir,
            outcome_name=outcome,
            n_splits_outer=n_splits_outer,
            n_repeats_outer=n_repeats_outer,
            n_splits_inner=n_splits_inner,
            seed=seed,
            parallel_mode=parallel_mode,
            n_jobs_outer=n_jobs_outer,
            blas_threads_per_worker=blas_threads_per_worker,
            lasso_n_jobs=lasso_n_jobs,
            use_memmap=use_memmap,
            mmap_cache_dir=None,
            keep_mmap_cache=keep_mmap_cache,
            compute_legacy_allcorr=compute_legacy_allcorr,
            loo_stride=loo_stride,
            inner_screen_mode=inner_screen_mode,
            eval_all_betas_on_test=eval_all_betas_on_test,
            consolidate_outputs=consolidate_outputs,
        )


if __name__ == "__main__":
    main()
