import os, sys
import numpy as np
import torch
sys.path.insert(0, "analysis")
from eval_signals import load_wm, wm_normalize, read_numeric_csv, auc_tie
GROUPS = {"lin_vel": (0, 3), "ang_vel": (3, 6), "gravity": (6, 9), "joint_pos": (9, 21), "joint_vel": (21, 33), "torque": (33, 45)}
def windows(state, action, idx, H):
    return np.stack([state[f - H:f] for f in idx]), np.stack([action[f - H + 1:f + 1] for f in idx])
@torch.no_grad()
def run_model(sd, state, action, idx, H, device, bs=512):
    feats, means = [], []
    for i in range(0, len(idx), bs):
        xs, xa = windows(state, action, idx[i:i + bs], H)
        xs = torch.from_numpy(xs).to(device); xa = torch.from_numpy(xa).to(device)
        sd.reset(); h = sd.state_base(xs, xa)
        m = torch.stack([head(h, xs)[0] for head in sd.state_heads])
        feats.append(h.cpu().numpy()); means.append(m.cpu().numpy())
    return np.concatenate(feats), np.concatenate(means, axis=1)
def probe_auc(X, y, env, folds=5, l2=1e-2, seed=0):
    rng = np.random.default_rng(seed); envs = np.unique(env); rng.shuffle(envs)
    score = np.zeros(len(y))
    for p in np.array_split(envs, folds):
        te = np.isin(env, p); tr = ~te
        if y[tr].sum() == 0 or y[te].sum() == 0: score[te] = np.nan; continue
        mu, sdv = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xt = torch.tensor((X[tr] - mu) / sdv, dtype=torch.float64); yt = torch.tensor(y[tr], dtype=torch.float64)
        w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True); b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        pw = (len(yt) - yt.sum()) / yt.sum()
        opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); z = Xt @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt, pos_weight=pw) + l2 * (w @ w)
            loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad(): score[te] = (torch.tensor((X[te] - mu) / sdv, dtype=torch.float64) @ w + b).numpy()
    ok = ~np.isnan(score)
    return auc_tie(score[ok & (y == 1)], score[ok & (y == 0)])
def main():
    trace, steps, sets_npz, out_txt = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    models = [a.split("=", 1) for a in sys.argv[5:]]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data = read_numeric_csv(trace); S = np.load(sets_npz); idx_all = S["idx"]
    pos_sets = {k: idx_all[S[f"in_{k}"]] for k in ("pre5", "lead10", "lead25") if f"in_{k}" in S.files}
    neg = idx_all[S["in_neg_clean"]]; lines = []
    def say(s=""): print(s, flush=True); lines.append(s)
    say(f"trace={trace} n_neg={len(neg)} " + " ".join(f"n_{k}={len(v)}" for k, v in pos_sets.items()))
    raw = data[:, 0:45]
    for k, pidx in pos_sets.items():
        ids = np.concatenate([pidx, neg]); y = np.r_[np.ones(len(pidx)), np.zeros(len(neg))]
        say(f"[raw-state probe] {k}: AUC={probe_auc(raw[ids - 1].astype(np.float64), y, ids // steps):.3f}")
    for name, ck in models:
        sd, ckpt, H = load_wm(ck, device)
        st, ac, _ = wm_normalize(data[:, 0:45].copy(), data[:, 45:57].copy(), ckpt)
        say(f"\n=== {name}  ({ck})")
        ids = np.unique(np.concatenate([*pos_sets.values(), neg]))
        feats, means = run_model(sd, st, ac, ids, H, device)
        pos_of = {f: i for i, f in enumerate(ids.tolist())}; sel = lambda a: np.array([pos_of[int(f)] for f in a])
        dstd = means.std(axis=0, ddof=1); epi = dstd.sum(axis=1)
        if "sig_epi" in S.files and os.path.basename(os.path.dirname(sets_npz)).startswith(name):
            say(f"sanity: max|epi - eval_signals epi| = {np.abs(epi - S['sig_epi'][np.searchsorted(idx_all, ids)]).max():.2e}")
        n_i = sel(neg)
        for k, pidx in pos_sets.items():
            p_i = sel(pidx); say(f"[epi] {k}: AUC={auc_tie(epi[p_i], epi[n_i]):.3f}  mean pos/neg={epi[p_i].mean():.3f}/{epi[n_i].mean():.3f}")
        p_i = sel(pos_sets["pre5"]); say("[groups] pre5   group: share_neg share_pre5 AUC")
        for g, (a, b) in GROUPS.items():
            gs = dstd[:, a:b].sum(1); say(f"   {g:9s}: {gs[n_i].sum()/epi[n_i].sum():.2f}  {gs[p_i].sum()/epi[p_i].sum():.2f}  {auc_tie(gs[p_i], gs[n_i]):.3f}")
        dim_auc = np.array([auc_tie(dstd[p_i, d], dstd[n_i, d]) for d in range(45)])
        say("[dims] top-5 per-dim AUC (pre5): " + ", ".join(f"d{d}={dim_auc[d]:.2f}" for d in np.argsort(-dim_auc)[:5]))
        W = [torch.cat([p.detach().flatten() for p in h.state_mean_layers.parameters()]).cpu().numpy() for h in sd.state_heads]
        cos = [np.dot(W[i], W[j]) / np.linalg.norm(W[i]) / np.linalg.norm(W[j]) for i in range(5) for j in range(i + 1, 5)]
        dev = means[:, n_i, :] - means[:, n_i, :].mean(0, keepdims=True)
        cor = [np.corrcoef(dev[i].ravel(), dev[j].ravel())[0, 1] for i in range(5) for j in range(i + 1, 5)]
        say(f"[heads] weight cosine mean={np.mean(cos):.3f}  | deviation corr on negatives mean={np.mean(cor):.3f}")
        for k, pidx in pos_sets.items():
            ids_k = np.concatenate([pidx, neg]); y = np.r_[np.ones(len(pidx)), np.zeros(len(neg))]
            say(f"[trunk probe] {k}: AUC={probe_auc(feats[sel(ids_k)].astype(np.float64), y, ids_k // steps):.3f}")
    open(out_txt, "w").write("\n".join(lines) + "\n")
main()
