# Graph Source Detection & Misinformation Blocking - Project Scoping

Status: pre-method scoping. Nothing below is a commitment to an approach. The goal and target venue must be fixed first (Section 0), then a problem (A, B, or C) chosen before algorithm design starts. The evaluation pipeline in this repository implements the baselines, datasets and metrics listed here so that any candidate method can be compared under one protocol.

---

## 0. Goal - decide before touching method

Two classic, related but distinct problems sit under "who started it / what do we do about it":

- **Problem A - Source Detection ("Patient Zero"):** given a snapshot of an epidemic/rumor on a known graph, infer the origin node(s).
- **Problem B - Influence/Misinformation Blocking (Containment):** given a "bad" cascade already spreading from a (possibly known) source, choose a limited set of nodes to block/counter it.
- **Problem C - candidate novel synthesis (unverified for prior art):** sequential/anytime-valid *detection* of "this is a malicious cascade" fused with a *blocking* decision, i.e. decide the stopping time to intervene with statistical (false-alarm-controlled) guarantees, then solve the blocking optimisation at that time. Structurally this is the SENTINEL-E recipe (anytime-valid e-process sequential testing + shift-robust conformal prediction + GNN multi-node coupling) applied to information diffusion instead of camera-based event detection. **Not confirmed novel - needs a dedicated literature pass (Section 5) before committing.**

Open questions to resolve before picking A/B/C and starting method design:

1. **Target venue / audience.** Pure algorithms + approximation guarantees (Algorithmica, ESA/ICALP-style) vs. applied data-mining/ML (KDD, WWW, CIKM, AAAI, TKDE) vs. sequential-statistics/ML-theory (if Problem C: NeurIPS/ICML/AISTATS or a stats journal) vs. domain-specific (public-health informatics / computational social science).
2. **What counts as the contribution?** A new approximation ratio / hardness result vs. a new practical algorithm beating SOTA on benchmarks vs. a new *framework* combining two known primitives (anytime-valid testing + submodular blocking) with its own guarantee.
3. **Domain framing.** Biological epidemic (SIR/SEIR on contact networks) vs. social-media misinformation (IC/LT/Hawkes cascades on social graphs). Literatures, baselines and datasets differ; covering both in one submission usually dilutes the contribution.
4. **Single vs. multi-source.** Multi-source is combinatorially harder (model selection) and much less mature: bigger gap, higher risk.
5. **Static vs. temporal network.** Almost all baselines assume a static, fully-known graph; temporal/streaming variants are recent (2023+) and thinner.

Workflow: pick answers to (1)-(5) first, write them at the top of a new `docs/METHOD.md`, then return to the baseline tables below to select the actual comparison set.

---

## Problem A - Source Detection ("Patient Zero")

### A.1 Formal definition (single-source, static graph)

Given a graph G(V, E) and an epidemic process (SI/SIR/SEIR, or IC/LT for social cascades) that starts at an unknown node v* at time t0, and an observed snapshot of node states at time t1 = t0 + T (T possibly known or unknown), infer v* (single-source) or a set S* of size k (multi-source). Output is a ranking / probability distribution over V.

### A.2 Baseline models, chronological (2010 -> 2026)

| Year | Method | Authors / Venue | Idea | In pipeline |
|---|---|---|---|---|
| 2010 | Rumor-center estimator | Shah & Zaman, *SIGMETRICS Perform. Eval. Rev.* 38(1):203-214 | ML source estimator on regular trees; BFS-tree heuristic for general graphs | `rumor` |
| 2011 | Rumor centrality (theory) | Shah & Zaman, *IEEE Trans. Info. Theory* 57(8):5163-5181 | Detectability bounds on trees | `rumor` |
| 2012 | Rumor Centrality estimator | Shah & Zaman, *SIGMETRICS 2012* | Generalises ML estimator | `rumor` |
| 2012 | NetSleuth | Prakash, Vreeken, Faloutsos, *ICDM 2012*; *KAIS* 38(1):35-59, 2014 | MDL selects number and identity of sources; Laplacian eigenvector ranking | `netsleuth` |
| 2012 | Observer-based localisation | Pinto, Thiran, Vetterli, *PRL* 109:068702 | Observers record arrival times; source from delay distribution | - (needs observers) |
| 2013 | Effective-distance | Brockmann & Helbing, *Science* 342(6164):1337-1342 | Metapopulation model; H1N1 case study | - |
| 2013 | Fast Monte-Carlo source localisation | Agaskar & Lu, SPIE 8858 | Sample arrival times for likelihood | - |
| 2014 | Dynamic Message Passing (DMP) | Lokhov, Mezard, Ohta, Zdeborova, *PRE* 90:012801 | BP approximation of posterior over sources | `dmp` |
| 2015 | Soft Margin Estimator (SME) | Antulov-Fantulin et al., *PRL* 114:248701 | Jaccard similarity of simulated vs observed outbreaks | `sme` |
| 2016 | Jordan-center estimator | Zhu & Ying, *IEEE/ACM ToN* 24(1):408-421 | Minimum eccentricity in infected subgraph; SIR | `jordan` |
| 2016 | Reconstructing an Epidemic over Time | Rozenshtein, Gionis, Prakash, Vreeken, *KDD 2016* | Steiner-tree cost | - |
| 2016 | Loopy BP for infection times | Braunstein & Ingrosso, *Sci. Rep.* 6:27538 | Posterior over infection times | - |
| 2018 | Fast spread-source detection | Paluch et al., *Sci. Rep.* 8(1):2508 | Observer-method speed-up | - |
| 2019 | Correlation-based observer estimator | Xu et al., *Physica A* 527:121267 | Correlation of arrival times vs distances | - |
| 2019 | GCNSI | Dong, Zheng, Hung, Su, Li, *CIKM 2019* | Multi-source GCN trained on simulated cascades | `gcn` |
| 2020 | GCN/GAT single-source | Shah, Dehmamy, Perra, Chinazzi, Barabasi, Vespignani, Yu, arXiv:2006.11913 | Detectability limit for cyclic graphs; GCN/GAT on simulated SIR/SEIR | `gcn_skip` |
| 2021 | MCGNN | Shu, Yu, Ruan, Zhang, Xuan | Two-channel GCN + line-graph edge features | - |
| 2021 | IGCN | Guo, Zhang, Zhang, Fu, *GLOBECOM 2021* | State-pair-conditioned aggregation | `igcn` |
| 2021 | SD-STGCN | Sha, Al Hasan, Mohler, *DSAA 2021* | Multiple snapshots | - |
| 2022 | VAE inverse problem | Ling, Jiang, Wang, Liang, *KDD 2022* | Observes per-node reach probabilities | - |
| 2023 | Active querying + MCMF | Sterchi, Hilfiker, Grutter, Bernstein, *Sci. Rep.* 13(1):11363 | Sequential/active-learning framing | `mcs` (MCMF likelihood) |
| 2023 | Backtracking Network / temporal GNN | Haddad & Figueiredo 2023; Ru, Moore, Zhang, Zeng, Yan, *AAAI 2023* | GNNs for temporal networks | - |
| 2024 | GIN-SD | Cheng, Zhu, Tang, Gao, Wang, *AAAI 2024* | Incomplete observation, positional encoding + attention | - |
| 2026 | Systematic benchmark | Sterchi, Brack, Hilfiker, arXiv:2512.20657 | Reproduces 4 GNN architectures vs traditional/MLP baselines; protocol adopted here. Code: github.com/martinSter/gnn-source-detection | protocol, `mlp`, `mcs` |

### A.3 Datasets

Empirical contact networks (epidemics simulated on top), shipped in `data/networks/`:

| Network | N | E | Source |
|---|---|---|---|
| Zachary's Karate Club | 34 | 77 | Zachary, *J. Anthropological Research* 33(4):452-473, 1977 |
| Iceland (sexual contact) | 75 | 114 | Haraldsdottir, Gupta, Anderson, *J. AIDS* 5(4):374-381, 1992 |
| Dolphin social network | 62 | 159 | Lusseau et al., *Behav. Ecol. Sociobiol.* 54(4):396-405, 2003 |
| Fraternity | 58 | 967 | Killworth & Bernard, *Human Organization* 35(3):269-286, 1976 |
| Workplace (static projection) | 92 | 755 | Genois et al., *Network Science* 3:326-347, 2015 |
| Highschool 2013 (static projection) | 327 | 5818 | Mastrandrea, Fournet, Barrat, *PLOS ONE* 10(9), 2015 |
| Powergrid | 4,941 | 6,594 | Watts & Strogatz 1998; used in Lokhov et al. 2014 |
| Air traffic (weighted, country level) | - | 2,317 | OpenFlights aggregation |

Real-world epidemic case study: 2009 H1N1 pandemic, country-level arrival times from Brockmann & Helbing 2013 (Table S4), OpenFlights airline network aggregated to country level.

Synthetic graphs: Erdos-Renyi, Barabasi-Albert, Watts-Strogatz, random trees (`er:N`, `ba:N:m=2`, `ws:N:k=4:p=0.1`, `tree:N`).

Social-media cascade datasets (real propagation trees; useful to calibrate IC/LT/Hawkes parameters, not blind source-detection benchmarks since the root is known): Twitter15/16 (Ma et al. 2017), PHEME (Zubiaga et al. 2016), Weibo (Ma et al. 2016), FakeNewsNet (Shu et al. 2018/2020).

### A.4 Metrics (Sterchi et al. 2026 standard set; all implemented in `trace_e/metrics.py`)

- Top-k accuracy (k = 1, 3, 5), ties broken uniformly at random.
- Average error distance (hops between MAP estimate and true source).
- Average reciprocal rank (average-rank tie handling).
- Average 90% credible-set size (probabilistic methods).
- Brier score and resistance score (graph-aware proper scoring rule; Hilfiker, Brack, Sterchi, manuscript 2026 - verify before citing).
- Detectability over time: top-k as a function of T (sweep `--T`).
- Runtime: simulation, training, per-instance inference.

### A.5 Known open gaps

1. Detectability limits proven mainly for trees; cyclic graphs lack tight theory.
2. Multi-source is a model-selection problem with no tight guarantee for "how many + which".
3. Incomplete/noisy observation only recently addressed (GIN-SD 2024), empirically.
4. Temporal/streaming networks are the newest sub-area (2023+).
5. Uncertainty in T and beta is rarely modelled; no method has calibrated coverage.
6. No existing method gives an anytime-valid confidence set for the source valid at any data-dependent stopping time - candidate gap for Problem C.

---

## Problem B - Influence / Misinformation Blocking (Containment)

### B.1 Formal definitions

**Influence Limitation (competitive cascade; Budak et al. 2011):** given a graph, a bad campaign seeded at a known set, and budget k, choose k good-campaign seeds to minimise the number of nodes ultimately adopting the bad campaign (first arrival wins).

**Influence Minimisation / vertex blocking (Xie et al. 2023/2024):** given a fixed bad seed set S and budget b, choose b vertices to remove so as to minimise the spread of S. NP-hard and hard to approximate.

### B.2 Baseline models, chronological (2011 -> 2024)

| Year | Method | Authors / Venue | Idea | In pipeline |
|---|---|---|---|---|
| 2011 | Influence Limitation | Budak, Agrawal, El Abbadi, *WWW 2011* | Multi-campaign IC; submodular greedy (1-1/e) | `greedy` (counter-seeding, CELF) |
| 2012 | Influence Blocking Maximisation | He, Song, Chen, Jiang, *SDM 2012* | Competitive LT | - |
| 2012 | Containment of misinformation | Nguyen, Yan, Thai, Eidenbenz, *WebSci 2012* | Node/edge removal | `greedy` (blocking mode) |
| 2013 | Negative Influence Minimising by blocking nodes | Wang et al., *AAAI Workshops* | Blocking formulation | `greedy` (blocking mode) |
| 2013 | Least Cost Rumor Blocking | Fan et al., *ICDCS 2013* | Cost-constrained | - |
| 2016 | DRIMUX | Wang et al., *AAAI 2016* | Dynamic rumor-influence minimisation | - |
| 2017 | Randomised rumor blocking | Tong et al., *INFOCOM 2017*; *IEEE TNSE* 2020 | (1-1/e-delta)-approx via reverse sampling | - (planned) |
| 2017 | Point-process intervention | Farajtabar et al., *ICML 2017* | Hawkes + RL | - |
| 2018 | Distributed rumor blocking | Tong, Wu, Du, *IEEE TCSS* | Multiple counter-cascades | - |
| 2020 | Minimising influence of rumors by blockers | Yan et al., *IEEE TNSE* 7(3) | Blocker placement | - |
| 2021 | Survey | Zareie & Sakellariou | Consolidates pre-2021 | - |
| 2023 | IMIN | Xie, Zhang, Wang, Lin, Zhang, *ICDE 2023* | NP-hard + hard to approximate; scalable heuristic | - (planned) |
| 2024 | Influence Minimisation via Blocking Strategies | Xie et al., *INFORMS J. Computing* | Code: github.com/INFORMSJoC/2024.0591 | - (planned) |
| 2024 | RBP | Xiang et al., *World Wide Web* | Pertinence-set constraint, HGF | - |
| 2024 | IM via Vertex Countering | Xie et al., *PVLDB* 17(6) | Dual problem | - |

Simple heuristics used as references in most of the above: random, degree, PageRank, proximity to seeds (`random`, `degree`, `pagerank`, `proximity`).

### B.3 Datasets

Standard IM/blocking graphs: NetHEPT / NetPHY (arXiv co-authorship), Flixster, Epinions, DBLP (SNAP), Twitter/Weibo follower graphs. Cascade data to calibrate edge probabilities: Twitter15/16, PHEME, Weibo, SNAP Higgs, Memetracker. Synthetic: ER, BA, WS, Kronecker. The contact networks in `data/networks/` are used for the first pass so both problems share graphs.

### B.4 Metrics

- Final misinformed fraction after intervention (direct objective).
- Saved fraction = 1 - (post-intervention spread / pre-intervention spread).
- Approximation ratio vs. optimum on small instances (brute force).
- Running time vs |V|, |E|, budget.
- Solution quality as a curve over budget.
- Robustness to diffusion-model misspecification (IC vs LT vs fitted).

### B.5 Known open gaps

1. All formulations assume the bad seed set is known when the blocking set is optimised; none address *when* to intervene with statistical confidence.
2. IMIN hardness-of-approximation for pure vertex removal; restricted classes with better approximation are open.
3. Point-process (Hawkes) and combinatorial IC/LT lines barely cite each other.
4. Distributional shift in edge activation probabilities is acknowledged but not tightly bounded.

---

## 5. Problem C - candidate direction: sequential (anytime-valid) detection-triggered blocking

Idea: fuse an anytime-valid sequential test (e-process / SPRT-style, controlling false-alarm probability at any data-dependent stopping time as the cascade streams in) with a submodular blocking-set optimisation, so the pipeline (i) decides *when* there is enough evidence to call a cascade malicious with a Type-I error guarantee and (ii) outputs a blocking/counter-seed set with an approximation guarantee at that stopping time.

Status: hypothesis, not a confirmed gap. Before committing:

- Dedicated literature search crossing "anytime-valid", "e-process", "sequential testing", "online/streaming" with "influence blocking", "rumor containment", "misinformation intervention".
- Check online/bandit IM (Wen et al. 2017 IMLinUCB; Li et al. 2020 NeurIPS; Perrault et al. 2020 ICML; Carpentier & Valko 2016) for confidence-sequence-style early stopping.
- Check Sterchi et al. 2023 active querying (closest sequential flavour on the detection side).

---

## Citation hygiene

- Hilfiker, Brack, Sterchi, "Graph-Aware Proper Scoring Rules for Node Selection Tasks" is listed as manuscript in preparation (2026) inside Sterchi et al. 2026; not independently verifiable yet.
- Zareie & Sakellariou 2021 survey: full journal name/DOI not confirmed in this scoping pass.
