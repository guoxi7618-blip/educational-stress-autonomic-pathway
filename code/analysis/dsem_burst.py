"""Bayesian dynamic structural equation model of the 14-day measurement bursts (framework step 7), fitted with NumPyro NUTS.

Daily variables: mean EMA stress, mean EMA negative affect, nocturnal lnRMSSD. Each person-wave has latent means nested in latent person means (latent mean centring).
Within a person-wave the variables follow a first-order vector autoregression, with that evening's stress and affect entering the same night's lnRMSSD.
The stress-to-night and night-to-next-day-affect slopes vary between persons. Missing days are integrated out by a Kalman filter.
Usage: python dsem_burst.py <processed_dir> <out_dir> [table_prefix]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import numpyro

numpyro.set_host_device_count(4)
numpyro.enable_x64()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro.diagnostics import summary  # noqa: E402
from numpyro.infer import MCMC, NUTS, init_to_value  # noqa: E402

EPS = 1e-8
LOG2PI = float(np.log(2 * np.pi))
NAMES = ["stress", "neg_affect", "ln_night_rmssd"]


def kalman_loglik(y, o, F, Q, P0):
    """Exact Gaussian likelihood of a VAR(1) with missing entries, by scalar measurement updates. y, o: (n, T, 3); F, Q, P0: (n, 3, 3)."""

    def step(carry, inp):
        m, P = carry
        yt, ot = inp
        ll = 0.0
        for d in range(3):
            s = P[:, d, d] + EPS
            v = yt[:, d] - m[:, d]
            K = P[:, :, d] / s[:, None]
            ll = ll + ot[:, d] * (-0.5 * (LOG2PI + jnp.log(s) + v**2 / s))
            m = m + (ot[:, d] * v)[:, None] * K
            P = P - (ot[:, d] * s)[:, None, None] * (K[:, :, None] * K[:, None, :])
        m = jnp.einsum("nij,nj->ni", F, m)
        P = F @ P @ jnp.swapaxes(F, 1, 2) + Q
        return (m, P), ll

    (_, _), lls = jax.lax.scan(step, (jnp.zeros(y.shape[::2]), P0), (jnp.swapaxes(y, 0, 1), jnp.swapaxes(o, 0, 1)))
    return lls.sum()


def model(y, o, person, n_person):
    m = numpyro.sample("mean", dist.Normal(0, 5).expand([3]).to_event(1))
    sd_b = numpyro.sample("sd_person", dist.HalfNormal(3).expand([3]).to_event(1))
    Lc = numpyro.sample("L_person", dist.LKJCholesky(3, 2.0))
    with numpyro.plate("person", n_person):
        mu_p = numpyro.sample("mu_person", dist.MultivariateNormal(m, scale_tril=sd_b[:, None] * Lc))
        z_a = numpyro.sample("z_a", dist.Normal(0, 1))
        z_b = numpyro.sample("z_b", dist.Normal(0, 1))
    tau = numpyro.sample("sd_wave", dist.HalfNormal(2).expand([3]).to_event(1))
    with numpyro.plate("session", y.shape[0]):
        mu_w = numpyro.sample("mu_wave", dist.Normal(mu_p[person], tau).to_event(1))

    lag = numpyro.sample("lag", dist.Normal(0, 1).expand([3, 3]).to_event(2))  # lag[to, from], order stress, affect, lnRMSSD
    a0 = numpyro.sample("a_stress_to_night", dist.Normal(0, 1))
    b0 = numpyro.sample("b_night_to_next_affect", dist.Normal(0, 1))
    s_a = numpyro.sample("sd_a", dist.HalfNormal(0.5))
    s_b = numpyro.sample("sd_b", dist.HalfNormal(0.5))
    c = numpyro.sample("c_affect_to_night", dist.Normal(0, 1))
    sd_e = numpyro.sample("sd_resid", dist.HalfNormal(2).expand([3]).to_event(1))
    rho = numpyro.sample("rho_stress_affect", dist.Uniform(-1, 1))

    a = a0 + s_a * z_a
    b = b0 + s_b * z_b
    A = jnp.broadcast_to(lag, (n_person, 3, 3)).at[:, 1, 2].set(b)
    G = jnp.broadcast_to(jnp.eye(3), (n_person, 3, 3)).at[:, 2, 0].set(a).at[:, 2, 1].set(c)  # (I - B0)^-1 is the same matrix for this lower-triangular B0
    S = jnp.array([[sd_e[0] ** 2, rho * sd_e[0] * sd_e[1], 0.0], [rho * sd_e[0] * sd_e[1], sd_e[1] ** 2, 0.0], [0.0, 0.0, sd_e[2] ** 2]])
    F = G @ A
    Q = G @ S @ jnp.swapaxes(G, 1, 2)
    P, Fk = Q, F  # stationary covariance by doubling
    for _ in range(6):
        P = P + Fk @ P @ jnp.swapaxes(Fk, 1, 2)
        Fk = Fk @ Fk
    numpyro.factor("y", kalman_loglik(y - mu_w[:, None, :], o, F[person], Q[person], P[person]))
    numpyro.deterministic("indirect_ab", a0 * b0)


def fit(y, o, person, n_person, warmup, samples, chains, seed, depth=8):
    """y is standardized; missing entries are NaN."""
    o_ = (~np.isnan(y)).astype(float)
    y_ = np.nan_to_num(y)
    mu_w = np.nan_to_num(np.nanmean(y, axis=1))
    mu_p = np.vstack([mu_w[person == i].mean(0) for i in range(n_person)])
    init = dict(mu_wave=mu_w, mu_person=mu_p, mean=mu_p.mean(0), sd_person=mu_p.std(0) + 0.1, sd_wave=np.full(3, 0.5), sd_resid=np.full(3, 0.8), lag=np.eye(3) * 0.1, a_stress_to_night=0.0, b_night_to_next_affect=0.0, c_affect_to_night=0.0, sd_a=0.05, sd_b=0.05, rho_stress_affect=0.2, z_a=np.zeros(n_person), z_b=np.zeros(n_person))
    k = NUTS(model, init_strategy=init_to_value(values={a: jnp.asarray(b) for a, b in init.items()}), max_tree_depth=depth, target_accept_prob=0.85, dense_mass=False)
    mc = MCMC(k, num_warmup=warmup, num_samples=samples, num_chains=chains, chain_method="parallel", progress_bar=False)
    mc.run(jax.random.PRNGKey(seed), jnp.asarray(y_), jnp.asarray(o_), jnp.asarray(person), n_person, extra_fields=("diverging", "num_steps"))
    return mc


def report(mc, sd_w):
    s = mc.get_samples(group_by_chain=True)
    d = summary({k: v for k, v in s.items() if k not in ("z_a", "z_b", "mu_person", "mu_wave", "L_person")}, group_by_chain=True)
    rows = []

    def add(name, x, block, rhat, neff, scale=1.0):
        x = np.asarray(x).reshape(-1)
        rows.append(dict(block=block, parameter=name, median=np.median(x), lo95=np.quantile(x, 0.025), hi95=np.quantile(x, 0.975), prob_direction=max((x > 0).mean(), (x < 0).mean()), unstd_median=np.median(x) * scale, rhat=float(rhat), ess=float(neff)))

    for k, nm, sc in (("a_stress_to_night", "当日压力→当夜 lnRMSSD", sd_w[2] / sd_w[0]), ("b_night_to_next_affect", "当夜 lnRMSSD→次日负性情绪", sd_w[1] / sd_w[2]), ("c_affect_to_night", "当日负性情绪→当夜 lnRMSSD", sd_w[2] / sd_w[1]), ("indirect_ab", "日内间接效应 a×b", sd_w[1] / sd_w[0])):
        add(nm, s[k], "固定路径", d[k]["r_hat"], d[k]["n_eff"], sc)
    for i, to in enumerate(NAMES):
        for j, fr in enumerate(NAMES):
            if (i, j) != (1, 2):  # the lnRMSSD-to-affect lag is the random slope above
                add(f"{fr}(t-1)→{to}(t)", s["lag"][:, :, i, j], "滞后路径", d["lag"]["r_hat"][i, j], d["lag"]["n_eff"][i, j], sd_w[i] / sd_w[j])
    add("sd(a) 个体间", s["sd_a"], "随机斜率", d["sd_a"]["r_hat"], d["sd_a"]["n_eff"])
    add("sd(b) 个体间", s["sd_b"], "随机斜率", d["sd_b"]["r_hat"], d["sd_b"]["n_eff"])
    add("压力与负性情绪残差相关", s["rho_stress_affect"], "残差", d["rho_stress_affect"]["r_hat"], d["rho_stress_affect"]["n_eff"])
    Lc = np.asarray(s["L_person"]).reshape(-1, 3, 3)
    R = Lc @ np.swapaxes(Lc, 1, 2)
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        add(f"个体均值相关 {NAMES[i]}–{NAMES[j]}", R[:, i, j], "个体间", float("nan"), float("nan"))
    icc = (np.asarray(s["sd_person"]) ** 2).reshape(-1, 3)
    tau2 = (np.asarray(s["sd_wave"]) ** 2).reshape(-1, 3)
    for i, nm in enumerate(NAMES):
        add(f"日间方差以外的个体占比 {nm}", icc[:, i] / (icc[:, i] + tau2[:, i]), "方差分解", float("nan"), float("nan"))
    T = pd.DataFrame(rows)
    z = {k: np.asarray(v) for k, v in s.items() if k in ("mu_wave", "mu_person", "z_a", "z_b")}
    allz = summary(z, group_by_chain=True)
    ex = mc.get_extra_fields(group_by_chain=True)
    diag = dict(max_rhat_hyper=float(T.rhat.max()), min_ess_hyper=float(T.ess.min()), max_rhat_latent=float(max(np.nanmax(v["r_hat"]) for v in allz.values())), divergences=int(np.asarray(ex["diverging"]).sum()), mean_steps=float(np.asarray(ex["num_steps"]).mean()))
    return T, diag


def main(data, out, prefix=""):
    data, out = Path(data), Path(out)
    W = pd.read_csv(data / f"{prefix}wearable_nightly.csv")
    E = pd.read_csv(data / f"{prefix}ema_prompts.csv")
    d = E[E.answered == 1].groupby(["pid", "wave", "burst_day"], as_index=False)[["stress", "neg_affect"]].mean()
    grid = W[["pid", "wave", "burst_day", "ln_night_rmssd"]].merge(d, on=["pid", "wave", "burst_day"], how="left").sort_values(["pid", "wave", "burst_day"])
    v = ["stress", "neg_affect", "ln_night_rmssd"]
    keep = grid.groupby(["pid", "wave"])[v].transform("count").sum(axis=1) > 0
    grid = grid[keep]
    mean_ = grid[v].mean()
    sd_w = (grid[v] - grid.groupby(["pid", "wave"])[v].transform("mean")).std()
    Z = (grid[v] - mean_) / sd_w
    days = int(grid.burst_day.max())
    keys = grid[["pid", "wave"]].drop_duplicates().reset_index(drop=True)
    y = np.full((len(keys), days, 3), np.nan)
    idx = {(a, b): i for i, (a, b) in enumerate(keys.itertuples(index=False))}
    for (a, b, c), r in zip(grid[["pid", "wave", "burst_day"]].itertuples(index=False), Z.to_numpy()):
        y[idx[(a, b)], c - 1] = r
    pid = pd.Categorical(keys.pid)
    person, n_person = np.asarray(pid.codes), len(pid.categories)
    print("person-waves", len(keys), "persons", n_person, "observed", np.round((~np.isnan(y)).mean((0, 1)), 3), flush=True)
    t0 = time.time()
    mc = fit(y, None, person, n_person, 1000, 1000, 4, 20261102)
    T, diag = report(mc, sd_w.to_numpy())
    diag.update(n_person=int(n_person), n_person_wave=len(keys), n_days=days, seconds=round(time.time() - t0), warmup=1000, samples_per_chain=1000, chains=4, within_sd=dict(zip(v, sd_w.round(4).tolist())), observed=dict(zip(v, np.round((~np.isnan(y)).mean((0, 1)), 4).tolist())))
    out.mkdir(parents=True, exist_ok=True)
    T.round(4).to_csv(out / "s38_dsem.csv", index=False, encoding="utf-8-sig")
    (out / "s38_dsem_diagnostics.json").write_text(json.dumps(diag, ensure_ascii=False, indent=1), encoding="utf-8")
    print(T.round(3).to_string())
    print(diag)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
