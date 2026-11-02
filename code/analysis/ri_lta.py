"""Random-intercept latent transition analysis (Muthen & Asparouhov, 2022) of the four crisis indicators across the four waves (framework step on subtypes).

Latent states follow a first-order Markov chain with one transition matrix. Each indicator has state-specific means held equal over waves, a residual variance and a loading on a person-level
random intercept with unit variance. CGI-S and any C-SSRS ideation enter through an ordered logit with thresholds shared by all states: a Gaussian indicator would spend extra states on the discrete scale points. Maximum likelihood integrates the intercept by Gauss-Hermite quadrature and the states by the forward algorithm, so missing indicators and waves stay in the likelihood.
With all loadings at zero the model is the ordinary LTA. Usage: python ri_lta.py <processed_dir> <out_dir> [table_prefix]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.scipy.special import logsumexp  # noqa: E402

NODES, WEIGHTS = np.polynomial.hermite.hermgauss(21)
NODES, LOGW = NODES * np.sqrt(2), np.log(WEIGHTS / np.sqrt(np.pi))
MIN_SD = 0.05
CATS = None  # number of ordered categories per indicator, 0 for continuous; set before fitting
BIN = None  # CATS > 0


def n_par(K, J, ri):
    return K * J + (J - int(BIN.sum())) + (J if ri else 0) + int(np.maximum(CATS - 2, 0).sum()) + (K - 1) + K * (K - 1)


def unpack(th, K, J, ri):
    i = 0
    nu = th[i:i + K * J].reshape(K, J); i += K * J
    Jc = J - int(BIN.sum())
    sd = jnp.ones(J).at[jnp.asarray(np.flatnonzero(~BIN))].set(MIN_SD + jnp.exp(th[i:i + Jc])); i += Jc
    if ri:
        lam = th[i:i + J]; i += J
    else:
        lam = jnp.zeros(J)
    tau = []
    for c in CATS:  # first threshold is 0 because every state has its own location
        m = max(int(c) - 2, 0)
        tau.append(jnp.concatenate([jnp.zeros(1), jnp.cumsum(jnp.exp(th[i:i + m]))]) if c else jnp.zeros(0)); i += m
    lp = jnp.concatenate([jnp.zeros(1), th[i:i + K - 1]]); i += K - 1
    logpi = lp - logsumexp(lp)
    off = th[i:i + K * (K - 1)].reshape(K, K - 1)
    rows = [jnp.insert(off[k], k, 0.0) for k in range(K)]
    tr = jnp.stack(rows)
    logT = tr - logsumexp(tr, axis=1, keepdims=True)
    return nu, sd, lam, logpi, logT, tau


def emissions(th, y, o, K, ri):
    """log p(y_it | state, node), shape (N, Q, T, K)."""
    J = y.shape[2]
    nu, sd, lam, _, _, tau = unpack(th, K, J, ri)
    q = jnp.asarray(NODES if ri else np.zeros(1))
    mean = nu[None, None, None, :, :] + (lam[None, :] * q[:, None])[None, :, None, None, :]
    yy = y[:, None, :, None, :]
    lp = -0.5 * (jnp.log(2 * jnp.pi) + 2 * jnp.log(sd) + ((yy - mean) / sd) ** 2)
    for j in np.flatnonzero(BIN):
        cuts = jnp.concatenate([jnp.array([-40.0]), tau[j], jnp.array([40.0])])
        cat = yy[..., j].astype(int)
        p = jax.nn.sigmoid(cuts[cat + 1] - mean[..., j]) - jax.nn.sigmoid(cuts[cat] - mean[..., j])
        lp = lp.at[..., j].set(jnp.log(p + 1e-12))
    return (lp * o[:, None, :, None, :]).sum(-1)


def forward(logpi, logT, logem):
    a = logpi + logem[:, :, 0]
    for t in range(1, logem.shape[2]):
        a = logsumexp(a[..., :, None] + logT, axis=-2) + logem[:, :, t]
    return logsumexp(a, axis=-1)


def nll(th, y, o, K, ri):
    J = y.shape[2]
    _, _, _, logpi, logT, _ = unpack(th, K, J, ri)
    ll_q = forward(logpi, logT, emissions(th, y, o, K, ri))
    lw = jnp.asarray(LOGW if ri else np.zeros(1))
    return -logsumexp(ll_q + lw, axis=1).sum()


def posterior(th, y, o, K, ri):
    """p(state_it | y_i), shape (N, T, K)."""
    J, T = y.shape[2], y.shape[1]
    _, _, _, logpi, logT, _ = unpack(th, K, J, ri)
    le = emissions(th, y, o, K, ri)
    al = [logpi + le[:, :, 0]]
    for t in range(1, T):
        al.append(logsumexp(al[-1][..., :, None] + logT, axis=-2) + le[:, :, t])
    be = [jnp.zeros_like(al[0])] * T
    for t in range(T - 2, -1, -1):
        be[t] = logsumexp(logT + le[:, :, t + 1, None, :] + be[t + 1][..., None, :], axis=-1)
    lw = jnp.asarray(LOGW if ri else np.zeros(1))
    ll = logsumexp(al[-1], axis=-1)
    node = ll + lw - logsumexp(ll + lw, axis=1, keepdims=True)
    g = jnp.stack([logsumexp(node[:, :, None] + al[t] + be[t] - ll[:, :, None], axis=1) for t in range(T)], axis=1)
    return np.asarray(jnp.exp(g))


def start(y, K, J, ri, rng):
    obs = y[~np.isnan(y).any(2)]
    centres = obs[rng.choice(len(obs), K, replace=False)]
    centres = np.where(BIN, (centres - (CATS - 1) / 2) * 2, centres)
    th = list((centres + rng.normal(0, 0.2, centres.shape)).ravel()) + list(np.log(np.nanstd(y, axis=(0, 1))[~BIN] * 0.7))
    if ri:
        th += list(rng.uniform(0.2, 0.8, J))
    th += [0.4] * int(np.maximum(CATS - 2, 0).sum())
    th += list(rng.normal(0, 0.3, K - 1))
    th += list(rng.normal(-1.5, 0.4, K * (K - 1)))
    return np.array(th, dtype=float)


def fit_k(y, K, ri, starts, seed):
    J = y.shape[2]
    o = jnp.asarray((~np.isnan(y)).astype(float))
    y0 = jnp.asarray(np.nan_to_num(y))
    f = jax.jit(jax.value_and_grad(lambda t: nll(t, y0, o, K, ri)))
    rng = np.random.default_rng(seed)
    best, lls = None, []
    fun = lambda t: tuple(np.asarray(v, dtype=float) for v in f(jnp.asarray(t)))
    for _ in range(starts):
        r = minimize(fun, start(y, K, J, ri, rng), jac=True, method="L-BFGS-B", options=dict(maxiter=800, maxfun=1500, ftol=1e-8))
        lls.append(-float(r.fun))
        if best is None or r.fun < best.fun:
            best = r
    lls = np.array(lls)
    return best.x, -float(best.fun), int((lls > lls.max() - 0.01).sum()), o, y0


def relabel(th, K, J, ri):
    """Order states by mean indicator level so that labels are stable across starts."""
    nu = np.asarray(th[:K * J]).reshape(K, J)
    return np.argsort(nu[:, ~BIN].mean(1))


def fit_table(y, Ks, starts, seed):
    N = len(y)
    J = y.shape[2]
    rows, fits = [], {}
    for ri in (False, True):
        for K in Ks:
            th, ll, rep, o, y0 = fit_k(y, K, ri, starts if K > 1 else 2, seed + K)
            p = n_par(K, J, ri)
            post = posterior(th, y0, o, K, ri)
            q = np.clip(post, 1e-12, 1)
            ent = 1 + (q * np.log(q)).sum() / (post.shape[0] * post.shape[1] * np.log(K)) if K > 1 else np.nan
            rows.append(dict(model="RI-LTA" if ri else "LTA", k=K, loglik=ll, npar=p, aic=-2 * ll + 2 * p, bic=-2 * ll + p * np.log(N), abic=-2 * ll + p * np.log((N + 2) / 24), entropy=ent, smallest_state=post.mean((0, 1)).min(), best_ll_replicated=rep))
            fits[(ri, K)] = th
            print(rows[-1], flush=True)
    return pd.DataFrame(rows), fits


def details(y, th, K, ri, names, seed=1):
    """State means, loadings, prevalence and transitions, with Monte Carlo intervals from the observed information."""
    J = y.shape[2]
    o = jnp.asarray((~np.isnan(y)).astype(float))
    y0 = jnp.asarray(np.nan_to_num(y))
    H = np.asarray(jax.hessian(lambda t: nll(t, y0, o, K, ri))(jnp.asarray(th)))
    V = np.linalg.pinv(H + np.eye(len(th)) * 1e-8)
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(th, (V + V.T) / 2, size=2000, method="svd")
    order = relabel(th, K, J, ri)
    ex = jax.vmap(lambda t: unpack(t, K, J, ri)[:5])(jnp.asarray(draws))
    est = unpack(jnp.asarray(th), K, J, ri)
    ci = lambda a: (np.quantile(a, 0.025, axis=0), np.quantile(a, 0.975, axis=0))
    Tm = np.exp(np.asarray(est[4]))[order][:, order]
    Td = np.exp(np.asarray(ex[4]))[:, order][:, :, order]
    lo, hi = ci(Td)
    trans = pd.DataFrame([dict(from_state=i + 1, to_state=j + 1, prob=Tm[i, j], lo95=lo[i, j], hi95=hi[i, j]) for i in range(K) for j in range(K)])
    nu = np.asarray(est[0])[order]
    nud = np.asarray(ex[0])[:, order]
    lo, hi = ci(nud)
    expit = lambda v: 1 / (1 + np.exp(-v))
    two = CATS == 2
    means = pd.DataFrame([dict(state=k + 1, indicator=names[j], scale="probability at mean intercept" if two[j] else ("logit location" if BIN[j] else "baseline z"), mean=expit(nu[k, j]) if two[j] else nu[k, j], lo95=expit(lo[k, j]) if two[j] else lo[k, j], hi95=expit(hi[k, j]) if two[j] else hi[k, j]) for k in range(K) for j in range(J)])
    lamd = np.asarray(ex[2])
    lo, hi = ci(lamd)
    load = pd.DataFrame(dict(indicator=names, loading=np.asarray(est[2]), lo95=lo, hi95=hi, resid_sd=np.where(BIN, np.nan, np.asarray(est[1]))))
    post = posterior(th, y0, o, K, ri)[:, :, order]
    return trans, means, load, post


def main(data, out, prefix=""):
    global BIN, CATS
    data, out = Path(data), Path(out)
    S = pd.read_csv(data / f"{prefix}scale_scores.csv")
    C = pd.read_csv(data / f"{prefix}clinician_assessment.csv")
    L = S[["pid", "wave", "phq9_total", "gad7_total"]].merge(C[["pid", "wave", "cgi_s", "cssrs_intensity"]], on=["pid", "wave"], how="outer")
    inds = ["phq9_total", "gad7_total", "cgi_s", "cssrs_intensity"]
    for c in inds[:2]:  # on the baseline scale, as the composites in main_models.R
        L[c] = (L[c] - L.loc[L.wave == 0, c].mean()) / L.loc[L.wave == 0, c].std()
    counts = L.cgi_s.value_counts().sort_index()
    print("CGI-S counts", counts.to_dict(), flush=True)
    keep = counts.index[counts >= 10]  # scale points with fewer than ten ratings join their neighbour
    L["cgi_s"] = L.cgi_s.clip(keep.min(), keep.max()).map({v: i for i, v in enumerate(sorted(keep))}).where(L.cgi_s.notna())
    L[inds[3]] = (L[inds[3]] > 0).astype(float).where(L[inds[3]].notna())
    CATS = np.array([0, 0, int(len(keep)), 2])
    BIN = CATS > 0
    pids = sorted(L.pid.unique())
    idx = {p: i for i, p in enumerate(pids)}
    y = np.full((len(pids), 4, 4), np.nan)
    for r in L.itertuples(index=False):
        y[idx[r.pid], int(r.wave)] = [getattr(r, c) for c in inds]
    tab, fits = fit_table(y, range(1, 7), 30, 20261102)
    out.mkdir(parents=True, exist_ok=True)
    tab.round(4).to_csv(out / "s36_rilta_fit.csv", index=False, encoding="utf-8-sig")
    ri = tab[tab.model == "RI-LTA"]
    k_best = int(ri.loc[ri.bic.idxmin(), "k"])
    rows = dict(k_bic_ri=k_best, k_bic_lta=int(tab[tab.model == "LTA"].set_index("k").bic.idxmin()), overall_best=[str(tab.loc[tab.bic.idxmin(), "model"]), int(tab.loc[tab.bic.idxmin(), "k"])], n=len(pids))
    for k, tag in ((k_best, "best"), (3, "k3")):
        tr, me, lo, post = details(y, fits[(True, k)], k, True, inds)
        tr.round(4).to_csv(out / f"s36_rilta_transitions_{tag}.csv", index=False, encoding="utf-8-sig")
        me.round(4).to_csv(out / f"s36_rilta_means_{tag}.csv", index=False, encoding="utf-8-sig")
        lo.round(4).to_csv(out / f"s36_rilta_loadings_{tag}.csv", index=False, encoding="utf-8-sig")
        st = post.argmax(2) + 1
        pd.DataFrame(st, columns=[f"state_w{t}" for t in range(4)]).assign(pid=pids).to_csv(out / f"s36_rilta_states_{tag}.csv", index=False)
        rows[f"prevalence_by_wave_{tag}"] = np.round(post.mean(0), 4).tolist()
        rows[f"stayers_all_waves_{tag}"] = round(float((st == st[:, [0]]).all(1).mean()), 4)
    (out / "s36_rilta_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(tab.round(2).to_string())
    print(rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
