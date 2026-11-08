# Submission log

## Networks (Wiley), submitted 2026-11-08

- Title: Adaptive Influence Minimization via Deferred Frontier Blocking
- Portal: https://authors.wiley.com/journal/NET (Free Format submission)
- Manuscript ID: (fill in from the confirmation email)
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
