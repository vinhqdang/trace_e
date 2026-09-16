"""Graph neural network source detectors trained on simulated outbreaks.

Implemented in plain PyTorch (sparse adjacency matmul) so no PyTorch Geometric
dependency is needed:

* ``gcn``      - stacked GCN layers, ReLU, per-node linear head, softmax over
                 nodes (the GCNSI / Dong et al. 2019 style architecture as
                 used in the Sterchi et al. benchmark).
* ``gcn_skip`` - input projection, GCN layers with batch-norm, leaky-ReLU and
                 residual connections (Shah et al. 2020 style).
* ``mlp``      - per-node MLP on the one-hot state, no message passing
                 (topology-blind control).
* ``igcn``     - state-pair-conditioned aggregation (Guo et al., GLOBECOM 2021):
                 nine scalar weights, one per (state_k, state_i) pair.

Training follows the benchmark protocol: 70/30 stratified split of the
simulated outbreaks, Adam (lr 1e-3, weight decay 5e-4), NLL loss, early
stopping on validation loss.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graphs import adjacency
from .base import SourceDetector, register


def normalised_adjacency(G, self_loops: bool = True) -> torch.Tensor:
    A = adjacency(G).tocoo()
    n = A.shape[0]
    rows, cols, vals = A.row, A.col, A.data
    if self_loops:
        rows = np.concatenate([rows, np.arange(n)])
        cols = np.concatenate([cols, np.arange(n)])
        vals = np.concatenate([vals, np.ones(n)])
    deg = np.bincount(rows, weights=vals, minlength=n)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    vals = dinv[rows] * vals * dinv[cols]
    idx = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.sparse_coo_tensor(idx, torch.tensor(vals, dtype=torch.float32), (n, n)).coalesce()


def spmm_batched(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A: sparse (n,n); x: (B,n,F) -> (B,n,F)."""
    B, n, Fd = x.shape
    y = torch.sparse.mm(A, x.permute(1, 0, 2).reshape(n, B * Fd))
    return y.reshape(n, B, Fd).permute(1, 0, 2)


class GCNNet(nn.Module):
    def __init__(self, n_in, hidden, layers, dropout, skip=False, batch_norm=False, activation="relu"):
        super().__init__()
        self.skip = skip
        self.dropout = dropout
        self.act = F.relu if activation == "relu" else F.leaky_relu
        self.inp = nn.Linear(n_in, hidden) if skip else None
        dims = [hidden if skip else n_in] + [hidden] * layers
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=skip) for i in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)]) if batch_norm else None
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, A):
        if self.inp is not None:
            x = self.inp(x)
        for i, lin in enumerate(self.lins):
            res = x
            h = spmm_batched(A, lin(x))
            if self.bns is not None:
                B, n, Fd = h.shape
                h = self.bns[i](h.reshape(B * n, Fd)).reshape(B, n, Fd)
            h = self.act(h)
            h = F.dropout(h, self.dropout, self.training)
            x = h + res if self.skip else h
        return F.log_softmax(self.head(x).squeeze(-1), dim=1)


class MLPNet(nn.Module):
    def __init__(self, n_in, hidden, layers, dropout):
        super().__init__()
        dims = [n_in] + [hidden] * layers
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layers)])
        self.dropout = dropout
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, A):
        for lin in self.lins:
            x = F.dropout(F.relu(lin(x)), self.dropout, self.training)
        return F.log_softmax(self.head(x).squeeze(-1), dim=1)


class IGCNNet(nn.Module):
    """Aggregation weight depends on the (source state, target state) pair of each edge."""

    def __init__(self, n_in, hidden, layers, dropout, edge_index: torch.Tensor, n_nodes: int):
        super().__init__()
        self.inp = nn.Linear(n_in, hidden)
        self.mus = nn.ParameterList([nn.Parameter(torch.ones(9) + 0.01 * torch.randn(9)) for _ in range(layers)])
        self.dropout = dropout
        self.head = nn.Linear(hidden, 1)
        self.register_buffer("src", edge_index[0])
        self.register_buffer("dst", edge_index[1])
        deg = torch.bincount(edge_index[1], minlength=n_nodes).clamp(min=1).float()
        self.register_buffer("deg", deg)

    def forward(self, x, A, states=None):
        # states: (B, n) integer node states
        B, n, _ = x.shape
        pair = states[:, self.src] * 3 + states[:, self.dst]  # (B, E)
        h = F.leaky_relu(self.inp(x))
        for mu in self.mus:
            w = mu[pair]  # (B, E)
            msg = w.unsqueeze(-1) * h[:, self.src, :]  # (B, E, H)
            agg = torch.zeros_like(h).index_add_(1, self.dst, msg) / self.deg.view(1, n, 1)
            h = F.dropout(F.leaky_relu(agg + h), self.dropout, self.training)
        return F.log_softmax(self.head(h).squeeze(-1), dim=1)


class _GNNBase(SourceDetector):
    probabilistic = True
    needs_training_sims = True
    arch = "gcn"

    def __init__(self, ctx, hidden=16, layers=5, dropout=0.1, lr=1e-3, weight_decay=5e-4, epochs=200,
                 patience=5, batch_size=128, device=None, verbose_every=10, log=None, **params):
        super().__init__(ctx, **params)
        self.hp = dict(hidden=hidden, layers=layers, dropout=dropout, lr=lr, weight_decay=weight_decay,
                       epochs=epochs, patience=patience, batch_size=batch_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.log = log
        self.verbose_every = verbose_every
        self.history = []

    def build(self):
        hp = self.hp
        if self.arch == "gcn":
            self.A = normalised_adjacency(self.ctx.G, self_loops=True)
            return GCNNet(3, hp["hidden"], hp["layers"], hp["dropout"])
        if self.arch == "gcn_skip":
            self.A = normalised_adjacency(self.ctx.G, self_loops=False)
            return GCNNet(3, hp["hidden"], hp["layers"], hp["dropout"], skip=True, batch_norm=True, activation="leaky_relu")
        if self.arch == "mlp":
            self.A = None
            return MLPNet(3, hp["hidden"], hp["layers"], hp["dropout"])
        if self.arch == "igcn":
            self.A = None
            A = adjacency(self.ctx.G).tocoo()
            ei = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
            return IGCNNet(3, hp["hidden"], hp["layers"], hp["dropout"], ei, self.ctx.n)
        raise ValueError(self.arch)

    def _forward(self, states_b: torch.Tensor):
        x = F.one_hot(states_b.long(), 3).float()
        if self.arch == "igcn":
            return self.model(x, None, states_b.long())
        return self.model(x, self.A)

    def fit(self, train=None):
        assert train is not None
        from sklearn.model_selection import train_test_split
        torch.manual_seed(self.ctx.seed)
        self.model = self.build().to(self.device)
        if self.A is not None:
            self.A = self.A.to(self.device)
        X = torch.tensor(train.states, dtype=torch.int8)
        y = torch.tensor(train.sources, dtype=torch.long)
        idx = np.arange(len(y))
        try:
            tr, va = train_test_split(idx, test_size=0.3, stratify=train.sources, random_state=42)
        except ValueError:
            tr, va = train_test_split(idx, test_size=0.3, random_state=42)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.hp["lr"], weight_decay=self.hp["weight_decay"])
        bs = self.hp["batch_size"]
        best, best_state, bad = float("inf"), None, 0
        t0 = time.time()
        rng = np.random.default_rng(self.ctx.seed)
        for ep in range(1, self.hp["epochs"] + 1):
            self.model.train()
            perm = rng.permutation(tr)
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                xb, yb = X[b].to(self.device), y[b].to(self.device)
                opt.zero_grad()
                loss = F.nll_loss(self._forward(xb), yb)
                loss.backward()
                opt.step()
            va_loss, va_acc = self._evaluate(X, y, va)
            self.history.append((ep, va_loss, va_acc))
            if self.log and (ep % self.verbose_every == 0 or ep == 1):
                self.log.info("[%s] epoch %d val_loss=%.4f val_acc=%.4f (%.0fs)", self.name, ep, va_loss, va_acc, time.time() - t0)
            if va_loss < best - 1e-9:
                best, bad = va_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.hp["patience"]:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.epochs_run = ep
        self.best_val_loss = best
        if self.log:
            self.log.info("[%s] trained %d epochs, best val_loss=%.4f, %.0fs", self.name, ep, best, time.time() - t0)

    @torch.no_grad()
    def _evaluate(self, X, y, idx):
        self.model.eval()
        loss, correct = 0.0, 0
        bs = 512
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            out = self._forward(X[b].to(self.device))
            yb = y[b].to(self.device)
            loss += F.nll_loss(out, yb, reduction="sum").item()
            correct += (out.argmax(1) == yb).sum().item()
        return loss / len(idx), correct / len(idx)

    @torch.no_grad()
    def score_batch(self, states):
        out = []
        bs = 512
        X = torch.tensor(states, dtype=torch.int8)
        for i in range(0, len(X), bs):
            out.append(self._forward(X[i:i + bs].to(self.device)).cpu().numpy())
        s = np.concatenate(out).astype(np.float64)
        s[states == 0] = -np.inf
        return s

    def score(self, states):
        return self.score_batch(states[None])[0]


@register
class GCN(_GNNBase):
    name = "gcn"
    arch = "gcn"


@register
class GCNSkip(_GNNBase):
    name = "gcn_skip"
    arch = "gcn_skip"


@register
class MLP(_GNNBase):
    name = "mlp"
    arch = "mlp"


@register
class IGCN(_GNNBase):
    name = "igcn"
    arch = "igcn"
