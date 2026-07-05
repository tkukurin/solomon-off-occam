# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy>=2.5.1",
# ]
# ///
"""mini mini experiments wip.

Kolmogorov -> Solomonoff -> PFNs -> domain priors.

1. computable mini-Solomonoff predictor: 
- mixture over a tiny program class,
- each program weighted 2^-(description length)
Prediction = posterior mixture.

2. mini-PFN: 
a small MLP meta-trained on tasks sampled from a prior.
min expected NLL over prior-sampled tasks makes the optimal net
posterior predictive. The net *amortizes* Bayes; no inference-time updating.

3. Domain restriction:
the same architecture trained on a narrow (domain) prior
beats the broad-prior net on in-domain tasks 
-> "approximations get better when you shrink the hypothesis space". 
K(env)-bits-of-regret.
"""

# everyone knows true randomness comes from how many 42s you put in the seed
import numpy as np; rng = np.random.default_rng(42_42_42)


# === 1. mini-Solomonoff induction on tiny program class ===

# program: repeat(pattern) for all binary patterns up to length L.
# Description length of repeat(p) ~ |p| bits  =>  prior weight 2^-|p|.
# Solomonoff prior M(x) = sum ( programs consistent with x of 2^-|p| )
# Predict next bit: M(x1)/M(x).

def programs(maxl: int):
  for L in range(1, maxl + 1):  # L = len(description in bits)
    for i in range(2 ** L):
      pat = [(i >> b) & 1 for b in range(L)]
      yield pat, L 


def solomonoff_predict(x, maxl=8):
  """P(next bit = 1 | x) under the mixture."""
  m0 = m1 = 0.0
  for pat, L in programs(maxl):
    # check consistency
    if all(x[t] == pat[t % len(pat)] for t in range(len(x))):
      w = 2.0 ** (-L)
      if pat[len(x) % len(pat)] == 1: m1 += w
      else: m0 += w
  return m1 / (m0 + m1)

print("=== 1. mini-Solomonoff repeat(011) ===")
# converges fast because the true generator is a SHORT program
# regret of the mixture ~ K(generator) bits total
seq = [0, 1, 1] * 6  # program: repeat(011)
for t in [1, 2, 3, 4, 6, 9, 12]:
  p = solomonoff_predict(seq[:t])
  print(f"after {t:2d} bits {seq[:t]}: P(next=1) = {p:.3f} (truth: {seq[t]})")


# === 2. mini-PFN amortized Bayesian posterior predictive ===

# Task family: coin, theta ~ prior.
# PFN: sample (theta, flips, next_flip) from the prior
# => train w/ logloss to predict next from summary of flips.
# => loss-minimizing function is the posterior predictive
# => nn implicitly learns Bayes

# for Beta(a,b) prior: P(head | h heads, t tails) = (h+a)/(n+a+b).

def sample_tasks(n_tasks, a, b, max_n=20, rng=rng):
  theta = rng.beta(a, b, n_tasks)
  n = rng.integers(0, max_n + 1, n_tasks)
  h = rng.binomial(n, theta)
  y = rng.binomial(1, theta)  # target = next flip
  X = np.stack([h, n - h, np.ones(n_tasks)], 1).astype(float)
  X[:, :2] /= (n[:, None] + 1) # feats = normalized counts
  return X, y.astype(float), h, n


class MLP:
  def __init__(self, d_in=3, d_h=32, rng=rng):
    s = rng.standard_normal
    self.W1 = s((d_in, d_h)) * 0.5; self.b1 = np.zeros(d_h)
    self.W2 = s((d_h, 1)) * 0.5;    self.b2 = np.zeros(1)
  def forward(self, X):
    self.H = np.tanh(X @ self.W1 + self.b1)
    return 1 / (1 + np.exp(-(self.H @ self.W2 + self.b2).ravel()))
  def train(self, X, y, lr=0.05, epochs=300, bs=512):
    for _ in range(epochs):
      idx = rng.permutation(len(y))
      for k in range(0, len(y), bs):
        i = idx[k:k + bs]
        p = self.forward(X[i]); d = (p - y[i])[:, None] / len(i)
        gW2 = self.H.T @ d; gb2 = d.sum(0)
        dH = d @ self.W2.T * (1 - self.H ** 2)
        gW1 = X[i].T @ dH; gb1 = dH.sum(0)
        self.W2 -= lr * gW2; self.b2 -= lr * gb2
        self.W1 -= lr * gW1; self.b1 -= lr * gb1

def nll(p, y):
  p = np.clip(p, 1e-9, 1 - 1e-9)
  return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

def bayes(h, n, a, b):
  return (h + a) / (n + a + b)

print("=== 2. mini-PFN recovers Bayes, Beta(1,1) ===")
Xtr, ytr, _, _ = sample_tasks(50_000, 1, 1)
net_broad = MLP(); net_broad.train(Xtr, ytr)
Xte, yte, h, n = sample_tasks(5_000, 1, 1)
print(f"net NLL   = {nll(net_broad.forward(Xte), yte):.4f}")
print(f"Bayes NLL = {nll(bayes(h, n, 1, 1), yte):.4f}  (Laplace rule (h+1)/(n+2))")
for hh, nn in [(0, 0), (3, 4), (7, 10)]:
    x = np.array([[hh / (nn + 1), (nn - hh) / (nn + 1), 1.0]])
    print(f"h={hh}, n={nn}: net={net_broad.forward(x)[0]:.3f}  bayes={bayes(hh, nn, 1, 1):.3f}")


# == 3. domain restriction makes the approximation better in-domain ===
# eg domain knowledge: in our business, coins are heads-biased: theta ~ Beta(10,2).
# Train the same net on narrow prior; evaluate both on in-domain tasks.
# amortizer wastes no mass (or capacity) on out-of-domain programs.

print("=== 3. broad vs domain prior, eval in-domain Beta(10,2) ===")
Xn, yn, _, _ = sample_tasks(50_000, 10, 2)
net_narrow = MLP(); net_narrow.train(Xn, yn)
Xte, yte, h, n = sample_tasks(5_000, 10, 2)
print(f"net(broad prior)  NLL = {nll(net_broad.forward(Xte), yte):.4f}")
print(f"net(domain prior) NLL = {nll(net_narrow.forward(Xte), yte):.4f}")
print(f"exact Bayes broad  NLL = {nll(bayes(h, n, 1, 1), yte):.4f}")
print(f"exact Bayes domain NLL = {nll(bayes(h, n, 10, 2), yte):.4f}")

