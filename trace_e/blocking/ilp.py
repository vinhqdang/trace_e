"""Exact scenario-based (sample-average) time-indexed ILP for influence minimisation.

At a decision point the controller samples ``K`` live-edge scenarios of the
*unobserved* part of the cascade (edges out of the current frontier and out of
inactive nodes; edges out of earlier active nodes have already been tested and
are dead), and solves

    min  (1/K) sum_k sum_v z[k, v, D_k]
    s.t. z[k, v, d] >= z[k, u, d-1] - (blocks of v before step d)   for live (u, v) in scenario k
         z[k, v, d] >= z[k, v, d-1]
         sum of blocking variables <= remaining budget              (per scenario)

where ``z[k, v, d]`` is 1 when v is active by step d in scenario k. This is the
time-indexed formulation of the static IMIN sample-average approximation, with
the blocking variables given a time index so the budget can be spent at
different steps.

Three controllers use it:

* ``ilp1``: the one-shot SAA optimum, solved once at time 0 (all budget now);
  the exact counterpart of the greedy one-shot planners.
* ``ilp_commit``: the same model re-solved every round, committing the whole
  plan each round (no deferral); the exact counterpart of COMMIT.
* ``ilp_roll``: a rolling-horizon staged model. Blocks "now" (``x``) are shared
  by all scenarios; later blocks (``y``, steps 1..H-1) are scenario specific
  recourse. Only the "now" blocks are committed, then the controller observes
  and re-solves. The scenario-specific recourse is optimistic (it assumes the
  future is known per scenario), which is the usual two-stage approximation of
  the multi-stage problem.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix


def _sample_scenarios(g, ep, K, rng):
    return _sample_tree(g, ep, K, rng, 0)[0]


def _sample_tree(g, ep, K, rng, M):
    """As above, but with ``M`` first-step branches (M = 0: every scenario is its own branch).

    Scenarios in one branch share the live edges out of the frontier, i.e. the
    information observed after the next step, so a decision taken after that
    step may be shared by the whole branch (non-anticipativity). Returns
    (scenarios, branch id per scenario).
    """
    if M <= 0:
        return _sample_core(g, ep, K, rng, 0)
    per = max(1, K // M)
    scen, bid = [], []
    for b in range(M):
        seed = int(rng.integers(2**31))
        sc, _ = _sample_core(g, ep, per, rng, seed)
        scen += sc
        bid += [b] * per
    return scen, bid


def _sample_core(g, ep, K, rng, layer1_seed):
    """Arrival times (BFS depth from the frontier, 1 = next step) in K sampled futures.

    Returns a list of (arrival dict v -> d, live-edge list [(u, v)]) restricted to
    edges that lie on shortest-or-longer live paths from the frontier through
    inactive, unblocked nodes.
    """
    inactive = ~(ep.active | ep.good | ep.blocked)
    frontier = [int(u) for u in ep.frontier]
    out = []
    for _ in range(K):
        l1 = np.random.default_rng(layer1_seed) if layer1_seed else None
        arr = {}
        edges = []
        # BFS over layers; the edges out of a node are sampled once, when it is reached
        cur = frontier
        depth = 0
        visited = set(frontier)
        while cur:
            depth += 1
            nxt = []
            for u in cur:
                lo, hi = g.indptr[u], g.indptr[u + 1]
                nb = g.indices[lo:hi]
                pr = g.weights[lo:hi]
                r_ = l1 if (l1 is not None and depth == 1) else rng
                live = r_.random(len(nb)) < pr
                for v in nb[live]:
                    v = int(v)
                    if not inactive[v]:
                        continue
                    edges.append((int(u), v))
                    if v not in visited:
                        visited.add(v)
                        arr[v] = depth
                        nxt.append(v)
            cur = nxt
        out.append((arr, edges))
    return out, list(range(K))


def solve_plan(g, ep, budget_left, K, H, staged, rng, time_limit, slack=4, M=0):
    """Solve the SAA ILP; return (nodes to block now, info dict)."""
    scen, bid = _sample_tree(g, ep, K, rng, M)
    cand = sorted({v for arr, _ in scen for v in arr})
    if not cand or budget_left <= 0:
        return [], dict(status="trivial", gap=0.0, nvars=0)
    xi = {v: i for i, v in enumerate(cand)}
    nv = len(cand)
    cols = nv  # running variable counter; x vars first
    lb, ub, cost, integ = [0.0] * nv, [1.0] * nv, [0.0] * nv, [1] * nv
    rows, cidx, vals, rhs_lo = [], [], [], []
    r = 0

    def newvar(c=0.0, integer=0):
        nonlocal cols
        lb.append(0.0)
        ub.append(1.0)
        cost.append(c)
        integ.append(integer)
        cols += 1
        return cols - 1

    budget_rows = []
    ybranch = {}
    for k, (arr, edges) in enumerate(scen):
        if not arr:
            continue
        # blocking can delay arrivals past the unblocked BFS depth; a path to a new node has at most |arr| steps
        Dk = min(max(arr.values()) + slack, len(arr))
        Dk = max(Dk, max(arr.values()))
        # recourse variables y[k, v, t], t = 1..min(H-1, d_k(v)-1)
        yv = {}
        if staged:
            for v in arr:
                for t in range(1, min(H - 1, Dk - 1) + 1):
                    if t == 1 and M > 0:  # decision after the first step: shared by the branch
                        if (bid[k], v) not in ybranch:
                            ybranch[(bid[k], v)] = newvar(0.0, 1)
                        yv[(v, t)] = ybranch[(bid[k], v)]
                    else:
                        yv[(v, t)] = newvar(0.0, 1)
        # z variables for d in [d_k(v), Dk]
        zv = {}
        for v, d in arr.items():
            for dd in range(d, Dk + 1):
                zv[(v, dd)] = newvar(0.0 if dd < Dk else 1.0 / K, 0)
        # incoming live edges
        inc = {}
        for u, v in edges:
            inc.setdefault(v, []).append(u)

        def blockterms(v, dd):
            t = [(xi[v], 1.0)]
            for tt in range(1, dd):
                if (v, tt) in yv:
                    t.append((yv[(v, tt)], 1.0))
            return t

        for v, d in arr.items():
            # monotone
            for dd in range(d + 1, Dk + 1):
                rows += [r, r]
                cidx += [zv[(v, dd)], zv[(v, dd - 1)]]
                vals += [1.0, -1.0]
                rhs_lo.append(0.0)
                r += 1
            for u in inc.get(v, []):
                du = arr.get(u, 0)  # frontier nodes have arrival 0 and are active
                for dd in range(max(d, du + 1), Dk + 1):
                    # z[v,dd] + blocks(v, <dd) >= z[u, dd-1]   (z[u, dd-1] = 1 if u is frontier)
                    terms = [(zv[(v, dd)], 1.0)] + blockterms(v, dd)
                    if u in arr:
                        terms.append((zv[(u, dd - 1)], -1.0))
                        lo = 0.0
                    else:
                        lo = 1.0
                    for c, a in terms:
                        rows.append(r)
                        cidx.append(c)
                        vals.append(a)
                    rhs_lo.append(lo)
                    r += 1
        # per-scenario budget (x shared, y scenario specific)
        brow = [(xi[v], 1.0) for v in cand] + [(c, 1.0) for c in yv.values()]
        budget_rows.append(brow)
    # budget rows are upper bounds: encode as  -sum >= -b
    for brow in budget_rows:
        for c, a in brow:
            rows.append(r)
            cidx.append(c)
            vals.append(-a)
        rhs_lo.append(-float(budget_left))
        r += 1
    A = coo_matrix((vals, (rows, cidx)), shape=(r, cols)).tocsr()
    res = milp(c=np.array(cost), constraints=LinearConstraint(A, np.array(rhs_lo), np.inf),
               integrality=np.array(integ), bounds=Bounds(np.array(lb), np.array(ub)),
               options=dict(time_limit=time_limit, mip_rel_gap=1e-4, disp=False))
    info = dict(status=int(res.status), nvars=cols, nrows=r)
    if res.x is None:
        info["gap"] = float("nan")
        return [], info
    info["gap"] = float(getattr(res, "mip_gap", float("nan")) or 0.0)
    info["obj"] = float(res.fun)
    info["timeout"] = res.status == 1
    x = res.x[:nv]
    return [cand[i] for i in np.flatnonzero(x > 0.5)], info


class ILPController:
    """mode in {"ilp1", "ilp_commit", "ilp_roll"}."""

    def __init__(self, mode, K=40, H=3, time_limit=20.0, seed=0, graph=None, slack=4, M=0):
        self.mode, self.K, self.H, self.tl, self.seed, self.g, self.slack, self.M = mode, K, H, time_limit, seed, graph, slack, M
        self.timeouts = 0
        self.solves = 0
        self.solve_time = 0.0

    def reset(self, budget):
        self.left = budget
        self.rng = np.random.default_rng(self.seed)
        self.done = False

    def act(self, ep, kappa=1.0):
        if self.left <= 0 or (self.mode == "ilp1" and self.done):
            return []
        self.done = True
        t0 = time.perf_counter()
        nodes, info = solve_plan(ep.g, ep, self.left, self.K, self.H, self.mode == "ilp_roll", self.rng, self.tl, self.slack, self.M)
        self.solve_time += time.perf_counter() - t0
        self.solves += 1
        self.timeouts += int(bool(info.get("timeout")))
        nodes = nodes[: self.left]
        self.left -= len(nodes)
        return nodes
