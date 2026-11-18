# Submission log

## Networks (Wiley), submitted 2026-11-08

- Title: Adaptive Influence Minimization via Deferred Frontier Blocking
- Portal: https://authors.wiley.com/journal/NET (Free Format submission)
- Manuscript ID: NET-26-01040
- Outcome: rejected by the handling editor (Dr. Archetti) on 2026-11-18 without external review. Reasons given: the writing is hard to follow at several places, and the contributions are summarized so compactly that the results cannot be evaluated.
- Files: `paper/networks/defer_twig_networks.tex`, PDF, and the LaTeX package `paper/networks/submission/defer_twig_networks_latex.zip`; cover letter source in `paper/networks/cover_letter/`
- Repository state at submission: `main`, commit `8f672ee` (manuscript and package); this note added afterwards
- Format: Wiley `USG.cls` template; abstract 245 words (limit 250); 7 keywords; numbered references; appendices kept inside the main file; the sample-journal logo and open-access badge were removed from the local copy of the class
- Statements made on the author's behalf: no conflicts of interest; no external funding; data availability points to https://github.com/vinhqdang/trace_e (the repository must be public)

### Previous submission

INFORMS Journal on Computing, JOC-2026-09-OA-2569, rejected 2026-11-06 by the area editor on the associate editor's assessment that the paper did not fit the journal's scope. Points raised:

1. The literature on exact (ILP-based) and heuristic influence minimization and rumor blocking was neglected, in the review and in the computational part. Suggested starting points: Tanınmış et al., EJOR 297(1), 2022, and Tanınmış et al., Computers & Operations Research 167, 2024.
2. Presentation: the introduction needed more accessible motivation, the order of material needed fixing (the worked example used Theorem 2 before it was stated), and the results needed intuition, not just lists of lemmas and theorems.

### What changed after the rejection

- Introduction rewritten with plain-language motivation and a roadmap; the worked example now introduces its purpose before the formal result; each result has an intuition paragraph
- Related work extended with exact and optimization-based methods
- New experiments (all on the same realizations for every policy, paired comparisons):
  - time-indexed scenario-based ILP, one-shot and staged, and the ILP as the planner inside DEFER (`scripts/run_ilp_baseline.sh`, `scripts/run_defer_ilp.sh`, `trace_e/blocking/ilp.py`, `tests/test_ilp.py`)
  - regime scan over five graph families (`scripts/run_regime_scan.sh`, `scripts/regime_report.py`)
  - star-of-paths benchmark (`scripts/star_of_paths_bench.py`)

### Results to keep in mind when answering reviewers

- Exact ILP planning did not beat greedy on small networks (within 0.02 of AG's saved fraction in 13 of 19 cells) and was far slower; on the 327-node network many solves hit the time limit, so those numbers are not exactly reproducible
- The staged ILP did not beat DEFER; DEFER wrapped around the ILP matched the ILP alone
- Re-plan-and-commit (COMMIT) was often as good as or better than DEFER on small instances
- Regime scan (45 cells): DEFER significantly better than AG in 9, than COMMIT in 6, never significantly worse; wins concentrated in large-cascade regimes; the grid was chosen after seeing the small-instance results, so it is exploratory
- Star-of-paths (18 cells): DEFER saved 0.36 to 0.97 of spread and beat the best of AG, GR, COMMIT and the one-shot ILP by at least 0.27 in every cell; this is a constructed family, not evidence of how common the structure is
- Not done: the cut-based Benders and bilevel algorithms of Tanınmış et al. were not implemented; the related-work text says so. Their descriptions rest on the titles only.

### Fallbacks if rejected

Computers & Operations Research (closer to the editor's feedback; expect demands for stronger computational comparison), Journal of Combinatorial Optimization or Discrete Applied Mathematics (theory-leaning; the paper would need cutting), INFORMS Journal on Optimization or Decision Analysis (scope unchecked).


## After the Networks decision: split into two papers (2026-11-18)

The combined manuscript (`paper/ijoc`, `paper/networks`) is superseded. It is replaced by two stand-alone papers in a neutral `article` format, to be formatted for a journal once chosen:

- `paper/defer/defer.tex`: adaptive influence minimization (DEFER). Summary-of-results table in the introduction; star-of-paths, regime scan, empirical networks and integer-programming baselines in the main text.
- `paper/twig/twig.tex`: exact spread evaluation on bounded-pathwidth graphs (TWIG) and the optimality gap of greedy blocking.

An independent review of both drafts found that several numbers in the combined manuscript could not be reproduced. Do not reuse the old versions. Corrections made:

- TWIG optimality gap: the old 56 of 60, 2.2% and 84.6% had no script behind them. The reproducible study gives 287 of 300 optimal, mean excess 0.32%, largest 44.8% (`scripts/twig_optimality_gap.py`). Timings and widths are logged by `scripts/twig_feasibility.py`; karate 2.5 s, Iceland 55 s, dolphin exceeded an 8 GB limit.
- DEFER empirical-network tables: the old tables mixed runs made with different versions of the code. They were regenerated from one run with the current code (`scripts/run_snap_benchmark.sh`, tables and figure by `scripts/snap_tables.py` and `scripts/snap_figure.py`). The new picture is weaker than the old one: DEFER significantly better than AG in 5 of 40 settings, never significantly worse; the largest gain is 13.9 points (ca-HepTh, constant probability, budget 20).
- Push-down does not help on the empirical networks (two borderline settings where DEFER is slightly worse than DEFER-nopush); on the star-of-paths family it is the whole advantage.
- Statements about other authors' papers were softened to what the repository supports (Nakamura and Nishino, Xie et al., Tanınmış et al.); AdvancedGreedy and GreedyReplace are credited to both Xie et al. papers.
