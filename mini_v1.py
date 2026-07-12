# %%
import io, contextlib
from math import lgamma
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
  from mini_v0 import MLP, sample_tasks, nll, bayes, rng, net_broad


# %% === 4. which nets do NOT recover the domain? ===

# two independent failure axes, parametrized:
#  (a) capacity: d_h too small => can't represent the posterior predictive surface
#  (b) sufficiency: features that destroy the sufficient statistic (h, n)
#      => NO width fixes it, the target isn't a function of the inputs anymore
# fun fact found the hard way: under Beta(1,1) the posterior mean (h+1)/(n+2) IS the
# smoothed ratio, so a ratio-only net secretly suffices there. the domain prior
# Beta(10,2) breaks the coincidence: (1,2) and (10,20) share ratio .5 but bayes
# says .786 vs .625. so we hunt failures ON the domain, as the question demands.
# metric: rmse(net probs, exact bayes probs) -- purer than nll (no label-noise floor)

def ratio_feats(h, n):
  return np.stack([(h + .5) / (n + 1), np.ones(len(h))], 1)  # ratio only, n discarded

def rmse_to_bayes(p_net, h, n, a, b):
  return np.sqrt(np.mean((p_net - bayes(h, n, a, b)) ** 2))

print("=== 4. failure modes on the domain Beta(10,2): capacity vs sufficiency ===")
Xtr, ytr, htr, ntr = sample_tasks(50_000, 10, 2)
Xte, yte, h, n = sample_tasks(20_000, 10, 2)
for w in [1, 2, 4, 32]:
  net = MLP(d_h=w); net.train(Xtr, ytr)
  print(f"d_h={w:2d}, feats=(h,t,1): rmse to bayes = {rmse_to_bayes(net.forward(Xte), h, n, 10, 2):.4f}")
net_ratio = MLP(d_in=2, d_h=32); net_ratio.train(ratio_feats(htr, ntr), ytr)
print(f"d_h=32, feats=ratio   : rmse to bayes = "
      f"{rmse_to_bayes(net_ratio.forward(ratio_feats(h, n)), h, n, 10, 2):.4f}  <- width can't fix this")
for hh, nn in [(1, 2), (10, 20)]:  # same ratio, different evidence mass
  p = net_ratio.forward(ratio_feats(np.array([hh]), np.array([nn])))[0]
  print(f"h={hh:2d}, n={nn:2d}: ratio-net={p:.3f}  bayes={bayes(hh, nn, 10, 2):.3f}")
# capacity failures shrink with width; the sufficiency failure does not (the ratio-net
# is FORCED to answer both rows identically). PFN moral: attention over raw examples
# exists so the net can COMPUTE its own sufficient statistics instead of being handed
# broken ones -- our summary features are training wheels that can also be a cage.


# %% === 5. more than one domain: mixture prior, latent vs oracle domain ===
# ref: Ortega et al. 2019, Meta-learning of Sequential Strategies -- arxiv.org/abs/1905.03030

# our business has two coin factories: heads-biased Beta(10,2), tails-biased Beta(2,10).
# z ~ Bernoulli(1/2) picks the factory, we never observe z (unless oracle).
# exact bayes = hierarchical: infer z from counts, then mix per-domain posteriors.
# marginal likelihood of counts under Beta(a,b) is beta-binomial: B(h+a, t+b) / B(a, b)

DOMS = [(10, 2), (2, 10)]

def logB(a, b): return lgamma(a) + lgamma(b) - lgamma(a + b)

def mix_bayes(h, n, doms=DOMS):
  lw = np.stack([[logB(hh + a, nn - hh + b) - logB(a, b) for hh, nn in zip(h, n)]
                 for a, b in doms])              # log evidence per domain
  lw -= lw.max(0); w = np.exp(lw); w /= w.sum(0) # posterior over factories
  preds = np.stack([bayes(h, n, a, b) for a, b in doms])
  return (w * preds).sum(0), w[0]                # predictive, P(factory A | data)

def sample_mix(n_tasks, doms=DOMS, max_n=20, rng=rng):
  z = rng.integers(0, 2, n_tasks)
  a = np.where(z == 0, doms[0][0], doms[1][0]); b = np.where(z == 0, doms[0][1], doms[1][1])
  theta = rng.beta(a, b)
  n = rng.integers(0, max_n + 1, n_tasks)
  h = rng.binomial(n, theta)
  y = rng.binomial(1, theta).astype(float)
  X = np.stack([h, n - h, np.ones(n_tasks)], 1).astype(float)
  X[:, :2] /= (n[:, None] + 1)
  return X, y, h, n, z

print("=== 5. two domains: net must infer the factory ===")
Xtr, ytr, _, _, ztr = sample_mix(50_000)
net_mix = MLP(); net_mix.train(Xtr, ytr, epochs=150)
net_orc = MLP(d_in=4)                            # oracle: gets z as a feature
net_orc.train(np.hstack([Xtr, ztr[:, None]]), ytr, epochs=150)
Xte, yte, h, n, z = sample_mix(20_000)
pm, wA = mix_bayes(h, n)
p_orc_bayes = np.where(z == 0, bayes(h, n, *DOMS[0]), bayes(h, n, *DOMS[1]))
print(f"net(mixture)       NLL = {nll(net_mix.forward(Xte), yte):.4f}")
print(f"exact mixture bayes NLL = {nll(pm, yte):.4f}")
print(f"net(oracle z)      NLL = {nll(net_orc.forward(np.hstack([Xte, z[:, None]])), yte):.4f}")
print(f"exact oracle bayes  NLL = {nll(p_orc_bayes, yte):.4f}")
for lo, hi in [(0, 2), (3, 8), (9, 20)]:
  m = (n >= lo) & (n <= hi)
  gap = nll(pm[m], yte[m]) - nll(p_orc_bayes[m], yte[m])
  print(f"n in [{lo:2d},{hi:2d}]: mixture-vs-oracle gap = {gap:.4f}  (cost of inferring z)")
# the gap is the price of the latent domain, <= H(z) = 1 bit, and it melts as n grows:
# data buys back the domain label. "which domain am i in" is itself bayesian inference,
# and the PFN amortizes THAT too, for free, just by training on the mixture.


# === 6. fine-tuning: who catastrophically forgets? ===
# ref: Kumar et al. 2022, Fine-Tuning Can Distort Pretrained Features (LP-FT) -- arxiv.org/abs/2202.10054

# start from net_broad (pretrained on Beta(1,1)), fine-tune on domain Beta(10,2).
# probe forgetting on the OPPOSITE region Beta(2,10): tasks the domain never shows.
# parametrized setups: full ft / head-only / replay / tiny lr.

import copy

def ft(net, X, y, lr=0.05, epochs=100, bs=512, head_only=False, rng=rng):
  net = copy.deepcopy(net)
  for _ in range(epochs):
    idx = rng.permutation(len(y))
    for k in range(0, len(y), bs):
      i = idx[k:k + bs]
      p = net.forward(X[i]); d = (p - y[i])[:, None] / len(i)
      net.W2 -= lr * (net.H.T @ d); net.b2 -= lr * d.sum(0)
      if not head_only:
        dH = d @ net.W2.T * (1 - net.H ** 2)
        net.W1 -= lr * (X[i].T @ dH); net.b1 -= lr * dH.sum(0)
  return net

print("=== 6. fine-tune broad -> Beta(10,2), probe forgetting on Beta(2,10) ===")
Xd, yd, _, _ = sample_tasks(5_000, 10, 2)         # small domain dataset, realistic
Xr, yr, _, _ = sample_tasks(1_000, 1, 1)          # replay buffer from pretraining prior
evals = {"in-domain B(10,2)": sample_tasks(20_000, 10, 2),
         "pretrain B(1,1)  ": sample_tasks(20_000, 1, 1),
         "probe    B(2,10) ": sample_tasks(20_000, 2, 10)}

setups = {
  "before ft        ": net_broad,
  "full ft          ": ft(net_broad, Xd, yd),
  "head-only ft     ": ft(net_broad, Xd, yd, head_only=True),
  "full ft + replay ": ft(net_broad, np.vstack([Xd, Xr]), np.concatenate([yd, yr])),
  "full ft, lr/25   ": ft(net_broad, Xd, yd, lr=0.002),
}
hdr = "  ".join(f"{k}" for k in evals); print(f"{'setup':<18} {hdr}   |w - w0|")
w0 = np.concatenate([p.ravel() for p in (net_broad.W1, net_broad.b1, net_broad.W2, net_broad.b2)])
for name, net in setups.items():
  row = "  ".join(f"{nll(net.forward(X), y):<17.4f}" for X, y, *_ in evals.values())
  wd = np.linalg.norm(np.concatenate([p.ravel() for p in (net.W1, net.b1, net.W2, net.b2)]) - w0)
  print(f"{name} {row}   {wd:.2f}")
for k, (X, y, h, n) in evals.items():
  a, b = {"in": (10, 2), "pr": (1, 1), "po": (2, 10)}[k[:2]]
  print(f"exact bayes on {k}: {nll(bayes(h, n, a, b), y):.4f}")
# what the table ACTUALLY shows (predictions were adjusted to reality, as is tradition):
#   full ft: best in-domain, worst on the probe -- it repainted the whole function
#   head-only: barely helps! folk wisdom says frozen features prevent forgetting, but a
#     32-wide head is expressive enough to repaint the function all by itself.
#     freezing protects only when the head is a bottleneck. parametrization != protection.
#   replay + tiny lr: what actually works. forgetting tracks the WEIGHT DRIFT column,
#     not which weights moved -- lr is a crude prior on "stay near pretrained", replay
#     keeps the old loss in view. this is why every lab mixes pretraining data back in.
# solomonoff reading: fine-tuning REWEIGHTS the amortized prior toward domain programs;
# forgetting = prior mass on out-of-domain programs leaking to zero. the fixes that work
# keep a spine of the broad prior in the objective (replay) or in weight space (small steps).


# === 7. architecture sweep: when does freezing actually protect? ===

# hypothesis from part 6: head-only ft failed to protect because a 32-wide head is
# expressive enough to repaint the function solo. so sweep d_h: the head is (d_h -> 1),
# freezing the body should protect exactly when the head is a BOTTLENECK.
# per width: pretrain on broad, then head-only vs full ft on the domain.
# report deltas vs that width's own pretrained net (fair across capacities):
#   adapt  = nll change in-domain  (more negative = better)
#   forget = nll change on probe   (more positive = worse)

print("=== 7. sweep d_h: freezing protects iff the head is a bottleneck ===")
Xb, yb, _, _ = sample_tasks(50_000, 1, 1)         # broad pretraining set
Xd, yd, _, _ = sample_tasks(5_000, 10, 2)         # domain ft set (same as part 6)
Xi, yi, _, _ = sample_tasks(20_000, 10, 2)        # in-domain eval
Xp, yp, _, _ = sample_tasks(20_000, 2, 10)        # forgetting probe

print(f"{'d_h':>3} {'head%':>6} | {'adapt(head)':>11} {'adapt(full)':>11} | {'forget(head)':>12} {'forget(full)':>12}")
for w in [1, 2, 4, 8, 32]:
  base = MLP(d_h=w); base.train(Xb, yb, epochs=150)
  i0, p0 = nll(base.forward(Xi), yi), nll(base.forward(Xp), yp)
  head = ft(base, Xd, yd, head_only=True)
  full = ft(base, Xd, yd)
  headpct = 100 * (w + 1) / (3 * w + w + w + 1)   # head params / total params
  print(f"{w:>3} {headpct:>5.0f}% | "
        f"{nll(head.forward(Xi), yi) - i0:>11.4f} {nll(full.forward(Xi), yi) - i0:>11.4f} | "
        f"{nll(head.forward(Xp), yp) - p0:>12.4f} {nll(full.forward(Xp), yp) - p0:>12.4f}")
# what it shows: forget(head) climbs monotonically with d_h (0.14 -> 0.28) and converges
# to full-ft forgetting at d_h=32 -- exactly the part-6 anomaly, now explained. freezing
# protects iff the trainable slice is a bottleneck. two footnotes the hypothesis missed:
#  (1) adaptation barely suffers even at d_h=1 -- THIS domain shift is mostly a
#      recalibration, so a 1-scalar head captures most of the gain. cheap protection.
#  (2) protection is partial everywhere: even one retrained scalar drags the probe
#      region along. safety by incapacity, not by design.
# the "freeze the backbone" folk remedy is a statement about architecture, not about
# fine-tuning -- it works when the trainable slice is low-rank wrt the task, which is
# the actual (often unstated) mechanism behind linear probes and small-r LoRA
# (Hu et al. 2021 -- arxiv.org/abs/2106.09685).


# === 8. bottleneck control: small SLICE vs small NET ===
# (no particular reference -- control experiment for parts 6/7; nearest neighbors are
#  LP-FT above and the low-rank-slice framing of LoRA / intrinsic dimensionality)

# part 7 confounds head size with network size: shrinking d_h shrank both. so: fix the
# body at 32 wide, insert a bottleneck, 3 -> 32 -> d_b -> 1, freeze everything up to
# and including the bottleneck, fine-tune only the (d_b -> 1) head. sweep d_b.
# if the freezing story is about the SLICE, forget(head) should track d_b while the
# pretrained baseline stays ~flat (capacity is carried by the 32-wide body).

class MLP2:  # 3 -> 32 -> d_b -> 1, the diet coke of deep learning
  def __init__(self, d_in=3, d_h=32, d_b=4, rng=rng):
    s = rng.standard_normal
    self.W1 = s((d_in, d_h)) * 0.5; self.b1 = np.zeros(d_h)
    self.Wb = s((d_h, d_b)) * 0.3;  self.bb = np.zeros(d_b)
    self.W2 = s((d_b, 1)) * 0.5;    self.b2 = np.zeros(1)
  def forward(self, X):
    self.H1 = np.tanh(X @ self.W1 + self.b1)
    self.Hb = np.tanh(self.H1 @ self.Wb + self.bb)
    return 1 / (1 + np.exp(-(self.Hb @ self.W2 + self.b2).ravel()))
  def step(self, X, y, lr, head_only=False):
    p = self.forward(X); d = (p - y)[:, None] / len(y)
    self.W2 -= lr * (self.Hb.T @ d); self.b2 -= lr * d.sum(0)
    if head_only: return
    dHb = d @ self.W2.T * (1 - self.Hb ** 2)
    self.Wb -= lr * (self.H1.T @ dHb); self.bb -= lr * dHb.sum(0)
    dH1 = dHb @ self.Wb.T * (1 - self.H1 ** 2)
    self.W1 -= lr * (X.T @ dH1); self.b1 -= lr * dH1.sum(0)
  def train(self, X, y, lr=0.05, epochs=150, bs=512, head_only=False, rng=rng):
    for _ in range(epochs):
      idx = rng.permutation(len(y))
      for k in range(0, len(y), bs): self.step(X[idx[k:k + bs]], y[idx[k:k + bs]], lr, head_only)
    return self

print("=== 8. fixed 32-wide body, sweep bottleneck d_b, ft only past it ===")
print(f"{'d_b':>3} {'base(in)':>8} | {'adapt(head)':>11} {'adapt(full)':>11} | {'forget(head)':>12} {'forget(full)':>12}")
for db in [1, 2, 4, 8, 32]:
  base = MLP2(d_b=db).train(Xb, yb)
  i0, p0 = nll(base.forward(Xi), yi), nll(base.forward(Xp), yp)
  head = copy.deepcopy(base).train(Xd, yd, epochs=100, head_only=True)
  full = copy.deepcopy(base).train(Xd, yd, epochs=100)
  print(f"{db:>3} {i0:>8.4f} | "
        f"{nll(head.forward(Xi), yi) - i0:>11.4f} {nll(full.forward(Xi), yi) - i0:>11.4f} | "
        f"{nll(head.forward(Xp), yp) - p0:>12.4f} {nll(full.forward(Xp), yp) - p0:>12.4f}")
# verdict: baseline in-domain nll is ~flat across d_b (capacity lives in the 32-wide
# body), yet forget(head) still climbs with d_b (0.16 -> 0.32) while forget(full)
# stays high everywhere. so part 7's effect was the SLICE, not the net: at fixed
# architecture, what you allow gradient to touch sets the forgetting budget.
# rank of the trainable slice ~ how many "directions" of the amortized prior you
# can overwrite. d_b is our r in LoRA-speak, chosen at architecture time.


# === 9. the bridge: head-only ⊂ LoRA ⊂ full ft, one dial ===
# ref: Hu et al. 2021, LoRA: Low-Rank Adaptation -- arxiv.org/abs/2106.09685

# the three regimes are one family: pick (which layers gradient touches) x (rank of
# the touched slice). head-only = rank-0 body + trained head. full ft = full-rank
# everywhere. LoRA-r = rank-r deltas W += A@B on body layers + trained head.
# B init to zero => ft starts exactly at the pretrained function, the polite way.
# we also log per-layer drift |dW|_F to watch WHERE each method spends its changes.

def lora_ft(net, X, y, r, lr=0.05, epochs=100, bs=512, rng=rng):
  net = copy.deepcopy(net)
  A1 = rng.standard_normal((net.W1.shape[0], r)) * .1; B1 = np.zeros((r, net.W1.shape[1]))
  Ab = rng.standard_normal((net.Wb.shape[0], r)) * .1; Bb = np.zeros((r, net.Wb.shape[1]))
  for _ in range(epochs):
    idx = rng.permutation(len(y))
    for k in range(0, len(y), bs):
      i = idx[k:k + bs]; Xi_, yi_ = X[i], y[i]
      W1e, Wbe = net.W1 + A1 @ B1, net.Wb + Ab @ Bb
      H1 = np.tanh(Xi_ @ W1e + net.b1); Hb = np.tanh(H1 @ Wbe + net.bb)
      p = 1 / (1 + np.exp(-(Hb @ net.W2 + net.b2).ravel()))
      d = (p - yi_)[:, None] / len(i)
      net.W2 -= lr * (Hb.T @ d); net.b2 -= lr * d.sum(0)  # head always trained
      dHb = d @ net.W2.T * (1 - Hb ** 2); g = H1.T @ dHb
      Ab -= lr * (g @ Bb.T); Bb -= lr * (Ab.T @ g)        # note: Ab already moved; sue us
      dH1 = dHb @ Wbe.T * (1 - H1 ** 2); g = Xi_.T @ dH1
      A1 -= lr * (g @ B1.T); B1 -= lr * (A1.T @ g)
  net.W1 += A1 @ B1; net.Wb += Ab @ Bb                    # merge, as the ancients intended
  return net

def drift(net, base):
  return [np.linalg.norm(a - b) for a, b in
          [(net.W1, base.W1), (net.Wb, base.Wb), (net.W2, base.W2)]]

print("=== 9. head-only / LoRA-r / full on MLP2(32 -> d_b=8) ===")
base9 = MLP2(d_b=8).train(Xb, yb)
i0, p0 = nll(base9.forward(Xi), yi), nll(base9.forward(Xp), yp)
nets9 = {
  "head-only (r=0)": copy.deepcopy(base9).train(Xd, yd, epochs=100, head_only=True),
  "lora r=1       ": lora_ft(base9, Xd, yd, r=1),
  "lora r=4       ": lora_ft(base9, Xd, yd, r=4),
  "full ft        ": copy.deepcopy(base9).train(Xd, yd, epochs=100),
}
print(f"{'method':<15} {'adapt':>8} {'forget':>8} | {'|dW1|':>6} {'|dWb|':>6} {'|dW2|':>6}")
for name, nt in nets9.items():
  a = nll(nt.forward(Xi), yi) - i0; f = nll(nt.forward(Xp), yp) - p0
  d1, db_, d2 = drift(nt, base9)
  print(f"{name:<15} {a:>8.4f} {f:>8.4f} | {d1:>6.3f} {db_:>6.3f} {d2:>6.3f}")
# one dial, two knobs: rank r interpolates head-only -> full in BOTH adapt and forget,
# and the drift columns show where the budget goes -- lora spreads small coherent
# nudges across layers, full ft repaints everything, head-only slams only W2.


# === 10. layer readouts: is the old domain erased, or just shadowed? ===
# ref: Ramasesh et al. 2020, Anatomy of Catastrophic Forgetting -- arxiv.org/abs/2007.07400

# forgetting measured at the OUTPUT confounds two stories: (a) the representation no
# longer carries the old domain's solution, vs (b) it's still in there, the readout
# just points elsewhere. disentangle with fresh linear probes: per layer, ridge-fit a
# new readout on frozen activations to predict the probe-domain EXACT bayes predictive,
# report rmse. low rmse = information survives at that depth; rising rmse = erasure.

def ridge_readout_rmse(acts_tr, t_tr, acts_te, t_te, lam=1e-3):
  A = np.hstack([acts_tr, np.ones((len(acts_tr), 1))])
  w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ t_tr)
  Ae = np.hstack([acts_te, np.ones((len(acts_te), 1))])
  return np.sqrt(np.mean((Ae @ w - t_te) ** 2))

def layer_acts(net, X):
  H1 = np.tanh(X @ net.W1 + net.b1); Hb = np.tanh(H1 @ net.Wb + net.bb)
  return {"x ": X, "H1": H1, "Hb": Hb}

Xp2, yp2, hp, np_ = sample_tasks(20_000, 2, 10)   # probe-domain tasks
t = bayes(hp, np_, 2, 10)                          # what a non-forgetful net should say
tr, te = slice(0, 15_000), slice(15_000, None)

print("=== 10. probe-domain bayes recoverable from layer activations? (rmse) ===")
print(f"{'method':<15} " + " ".join(f"{k:>7}" for k in ["x ", "H1", "Hb"]) + "   out-nll")
for name, nt in {"before ft      ": base9, **nets9}.items():
  acts = layer_acts(nt, Xp2)
  row = " ".join(f"{ridge_readout_rmse(acts[k][tr], t[tr], acts[k][te], t[te]):>7.4f}"
                 for k in acts)
  print(f"{name:<15} {row}   {nll(nt.forward(Xp2), yp2):.4f}")
# how to read it: the x-column is the same for everyone (linear model on raw feats,
# the floor of ignorance). if H1/Hb rmse stays at before-ft levels while out-nll blows
# up, the old domain is SHADOWED (readout-level forgetting, cheaply reversible by
# retraining a head). if Hb rmse itself climbs, the representation was overwritten --
# true erasure, and it should climb most for full ft, least for head-only/low-r lora,
# i.e. forgetting lives in the layers gradient was allowed to touch.
# verdict from the run: out-nll blows up for EVERYONE (0.47 -> 0.66..0.83) yet H1
# readout rmse is untouched for all methods (0.0049 -> 0.0051 even for full ft), and
# Hb climbs only for full ft (0.0074 -> 0.0104). so ~all of the forgetting here is
# shadowing, not erasure: the broad prior's solution is still linearly sitting in the
# activations, the network just stopped reporting it. erasure begins only at the
# deepest touched layer, only for full ft. which reframes part 6: "catastrophic"
# forgetting in the readout, mild amnesia in the representation -- and predicts the
# cheap fix: to un-forget, retrain a head on frozen features (see: how we built the
# probes). solomonoff reading: the amortized prior's program library survives ft
# largely intact; ft mostly rewires which programs the readout listens to.


# === 11. erasure timing: when does shadowing tip over? + nonlinear probes ===
# (no particular reference -- follow-up control for part 10)

# two follow-ups part 10 begs for:
#  (a) track out-nll and Hb-readout rmse DURING full ft: shadowing should be instant
#      (readout rewires in the first epochs), erasure gradual (representation drifts).
#  (b) linear probes lower-bound what's recoverable. maybe full ft's Hb still holds the
#      old solution, just non-linearly. probe with a small MLP and see how much comes back.

print("=== 11a. full ft on MLP2(32,8): out-nll vs Hb-readout rmse over epochs ===")
net11 = copy.deepcopy(base9); done = 0
print(f"{'epochs':>6} {'out-nll':>8} {'Hb rmse':>8}")
for ckpt in [0, 5, 10, 20, 50, 100]:
  net11.train(Xd, yd, epochs=ckpt - done); done = ckpt
  acts = layer_acts(net11, Xp2)
  r = ridge_readout_rmse(acts["Hb"][tr], t[tr], acts["Hb"][te], t[te])
  print(f"{ckpt:>6} {nll(net11.forward(Xp2), yp2):>8.4f} {r:>8.4f}")

print("=== 11b. linear vs mlp probe on Hb (probe-domain bayes target) ===")
for name, nt in [("before ft", base9), ("full ft  ", nets9["full ft        "])]:
  acts = layer_acts(nt, Xp2)["Hb"]
  lin = ridge_readout_rmse(acts[tr], t[tr], acts[te], t[te])
  mp = MLP(d_in=acts.shape[1], d_h=16); mp.train(acts[tr], t[tr], epochs=80)
  mlp_r = np.sqrt(np.mean((mp.forward(acts[te]) - t[te]) ** 2))
  print(f"{name}: linear rmse = {lin:.4f}   mlp rmse = {mlp_r:.4f}")
# verdict from the run: the mlp probe came out WORSE than ridge (0.026 vs 0.007..0.010)
# -- at 8 dims, closed-form ridge is simply the better estimator, i.e. the linear probe
# already recovers ~everything recoverable and the "folded, not destroyed" hypothesis
# has no gap left to close at this scale. probes lower-bound information by the probe's
# own optimization quality; do not read a bad probe as erasure (we almost did).
# 11a's verdict is the crisp one: out-nll takes ~60% of its total damage in the first
# 10 epochs while Hb rmse has moved ~20% of its eventual rise -- shadowing is a step
# function, erasure is a slow leak. early stopping mostly buys back the readout.


# === 12. a transformer: attention invents its own sufficient statistics ===
# ref: Muller et al. 2022, Transformers Can Do Bayesian Inference -- arxiv.org/abs/2112.10510

# everything so far spoon-fed (h, t) summaries. real PFNs eat RAW examples. so: raw
# flip sequences [BOS, x1..xn, PAD..], one attention head with a learned query, tiny
# ffn, manual backprop because torch is for people with deadlines. exchangeable data
# => no positional encodings, on purpose.

MAXN = 12; SEQT = MAXN + 1; BOS, PADT = 2, 3

def sample_seq(n_tasks, a, b, rng=rng):
  theta = rng.beta(a, b, n_tasks); n = rng.integers(0, MAXN + 1, n_tasks)
  flips = (rng.random((n_tasks, MAXN)) < theta[:, None]).astype(int)
  ids = np.full((n_tasks, SEQT), PADT); ids[:, 0] = BOS
  m = np.arange(MAXN)[None, :] < n[:, None]; ids[:, 1:][m] = flips[m]
  y = (rng.random(n_tasks) < theta).astype(float)
  return ids, y, (flips * m).sum(1), n

def seq_mask(n): pos = np.arange(SEQT)[None, :]; return (pos == 0) | (pos <= n[:, None])

class TF:  # one head, one block, zero positions. a transformer the way a haiku is a poem
  def __init__(self, d=8, dm=16, rng=rng):
    s = rng.standard_normal
    self.E = s((4, d)) * .5; self.Wk = s((d, d)) * .3; self.Wv = s((d, d)) * .3
    self.q = s(d) * .3; self.Wh = s((d, dm)) * .3; self.bh = np.zeros(dm)
    self.w = s(dm) * .3; self.b = np.zeros(1)
    self.P = [self.E, self.Wk, self.Wv, self.q, self.Wh, self.bh, self.w, self.b]
    self.M = [np.zeros_like(p) for p in self.P]
  def forward(self, ids, mask):
    e = self.E[ids]; k = e @ self.Wk; v = e @ self.Wv
    s = np.where(mask, k @ self.q, -1e9); s -= s.max(1, keepdims=True)
    a = np.exp(s); a /= a.sum(1, keepdims=True)
    pooled = (a[:, :, None] * v).sum(1)
    hh = np.tanh(pooled @ self.Wh + self.bh)
    self.c = (e, k, v, a, pooled, hh, ids, mask)
    return 1 / (1 + np.exp(-(hh @ self.w + self.b)))
  def step(self, ids, mask, y, lr, mom=.9):
    p = self.forward(ids, mask); e, k, v, a, pooled, hh, ids, mask = self.c
    dp = (p - y) / len(y)
    gw = hh.T @ dp; gb = np.array([dp.sum()])
    dh = dp[:, None] * self.w[None, :] * (1 - hh ** 2)
    gWh = pooled.T @ dh; gbh = dh.sum(0); dpool = dh @ self.Wh.T
    dv = a[:, :, None] * dpool[:, None, :]
    da = (v * dpool[:, None, :]).sum(2)
    ds = np.where(mask, a * (da - (a * da).sum(1, keepdims=True)), 0.)
    gq = (ds[:, :, None] * k).sum((0, 1)); dk = ds[:, :, None] * self.q[None, None, :]
    gWk = np.einsum('btd,bte->de', e, dk); gWv = np.einsum('btd,bte->de', e, dv)
    de = dk @ self.Wk.T + dv @ self.Wv.T
    gE = np.zeros_like(self.E); np.add.at(gE, ids.ravel(), de.reshape(-1, de.shape[-1]))
    for P, M, g in zip(self.P, self.M, [gE, gWk, gWv, gq, gWh, gbh, gw, gb]):
      M *= mom; M += g; P -= lr * M
  def train(self, ids, mask, y, lr=.3, epochs=30, bs=512, rng=rng):
    for _ in range(epochs):
      idx = rng.permutation(len(y))
      for kk in range(0, len(y), bs): self.step(ids[idx[kk:kk+bs]], mask[idx[kk:kk+bs]], y[idx[kk:kk+bs]], lr)
    return self

print("=== 12. transformer on raw sequences, broad prior ===")
ids, ytf, _, n_ = sample_seq(40_000, 1, 1); msk = seq_mask(n_)
tf12 = TF().train(ids, msk, ytf)
ide, ye, he, ne = sample_seq(10_000, 1, 1); mke = seq_mask(ne)
print(f"tf NLL = {nll(tf12.forward(ide, mke), ye):.4f}   bayes NLL = {nll(bayes(he, ne, 1, 1), ye):.4f}")
demo = np.full((1, SEQT), PADT); demo[0, 0] = BOS; demo[0, 1:6] = [1, 0, 1, 1, 0]
p = tf12.forward(demo, seq_mask(np.array([5])))[0]
print(f"seq 10110: tf={p:.3f}  bayes(h=3,n=5)={bayes(3, 5, 1, 1):.3f}")
print(f"attention: {np.round(tf12.c[3][0][:7], 3)}  <- ~uniform over flips, extra mass on BOS")
# the model was never told about (h, n). it learned to attend ~uniformly over flips
# (= counting) and parks extra mass on BOS, whose value vector acts as the pseudocount
# anchor -- a soft reinvention of laplace smoothing, because that's what the prior
# demanded. part 4's moral, demonstrated: given raw examples, attention BUILDS the
# sufficient statistic we were previously hand-feeding.
# then the same ft story, on sequences:
idd, ydd, _, nd = sample_seq(5_000, 10, 2)
tf_ft = copy.deepcopy(tf12).train(idd, seq_mask(nd), ydd, epochs=15)
idp, ypp, hp2, np2 = sample_seq(10_000, 2, 10); mkp = seq_mask(np2)
idi, yii, hi2, ni2 = sample_seq(10_000, 10, 2); mki = seq_mask(ni2)
print(f"ft on B(10,2): in-domain {nll(tf12.forward(idi, mki), yii):.4f} -> {nll(tf_ft.forward(idi, mki), yii):.4f}"
      f"  (bayes {nll(bayes(hi2, ni2, 10, 2), yii):.4f})")
print(f"probe B(2,10): {nll(tf12.forward(idp, mkp), ypp):.4f} -> {nll(tf_ft.forward(idp, mkp), ypp):.4f}"
      f"  (bayes {nll(bayes(hp2, np2, 2, 10), ypp):.4f})  <- same disease, new species")


# === 13. undercapacitated twin: a whole net the size of the trainable slice ===
# ref: Mirzadeh et al. 2022, Wide Neural Networks Forget Less Catastrophically -- arxiv.org/abs/2110.11526

# lora r=1 on MLP2(32,8) trains ~84 params inside a ~400-param frozen body. question:
# is the frozen body doing anything, or is 84 trainable params just 84 params? build
# an MLP2(d_h=8, d_b=4): ~81 params TOTAL. same experiment: pretrain broad, full ft
# on domain, adapt/forget + part-10 readouts. if shadowing needs slack capacity, the
# twin should show more genuine erasure -- nowhere to hide the old function.

print("=== 13. MLP2(8,4), 81 params total ~ lora r=1 slice (84 trainable) ===")
tw = MLP2(d_h=8, d_b=4).train(Xb, yb)
i0, p0 = nll(tw.forward(Xi), yi), nll(tw.forward(Xp), yp)
tw_ft = copy.deepcopy(tw).train(Xd, yd, epochs=100)
print(f"base(in) = {i0:.4f}  (big-net base was 0.47ish; 81 params still ~enough here)")
print(f"adapt = {nll(tw_ft.forward(Xi), yi) - i0:.4f}   forget = {nll(tw_ft.forward(Xp), yp) - p0:.4f}"
      f"   (lora r=1 on big net: -0.035 / +0.19)")
print(f"{'method':<15} " + " ".join(f"{k:>7}" for k in ['x ', 'H1', 'Hb']) + "   out-nll")
for name, nt in [("twin before ft ", tw), ("twin full ft   ", tw_ft)]:
  acts = layer_acts(nt, Xp2)
  row = " ".join(f"{ridge_readout_rmse(acts[k][tr], t[tr], acts[k][te], t[te]):>7.4f}" for k in acts)
  print(f"{name:<15} {row}   {nll(nt.forward(Xp2), yp2):.4f}")
# verdict: slice-matched comparison, ~84 trainable params each way:
#   lora r=1 in frozen 400-param body: adapt -0.035, forget +0.19, old-domain Hb rmse 0.0076
#   twin, 81 params total, full ft:    adapt -0.032, forget +0.37, old-domain Hb rmse 0.0175
# same baseline quality, same adaptation, DOUBLE the forgetting, and the old solution is
# >2x less recoverable from the twin's deepest layer -- note its rmse was already 0.0166
# BEFORE ft: an at-capacity representation has no slack in which to keep a solution it
# isn't currently using. the frozen body isn't dead weight; it's the archive. width
# buys forgetting-resistance not by learning more but by having room to not-unlearn.
