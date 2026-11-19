import re, sys, os, shutil
sys.path.insert(0, '.')
import orpbib
SRC = '/home/user/trace_e/paper/defer/defer.tex'
OUT = '/home/user/trace_e/paper/orp-defer'
t = open(SRC).read()

title = re.search(r"\\title\{(.*?)\}", t).group(1)
body_i = t.index("\\section{Introduction}"); bib_i = t.index("\\begin{thebibliography}"); bib_j = t.index("\\end{thebibliography}") + len("\\end{thebibliography}")
app_i = t.index("\\appendix"); end_i = t.index("\\end{document}")
main = t[body_i:bib_i].rstrip() + "\n"; appendix = t[app_i + len("\\appendix"):end_i].strip() + "\n"
bibs = orpbib.build(t)

ABSTRACT = r"""Blocking the spread of a rumor or epidemic that has already started, by removing a few nodes under a fixed budget, is usually posed as a one-shot decision made before the outbreak is observed. We ask whether spending the budget gradually, as the cascade unfolds, can be worse than spending it at once, and how large the benefit can be. We introduce DEFER, a wrapper that turns any one-shot influence minimization planner into an adaptive one: it re-plans every round and blocks a planned node only once the cascade has reached its neighborhood. We prove that the pure-deferral variant is never worse in expectation than the planner it wraps, provided plan replacement passes a held-out test whose residual error we do not bound; that a local rule for early commitment can fail when deferred targets compete for the budget; and that the adaptivity gap of influence minimization is unbounded. With every policy facing the same random cascades, DEFER saves 36 to 97 percent of the spread on the star-of-paths family used in the proof and beats the greedy, re-planning and one-shot integer-programming policies in all 18 settings. On 45 synthetic settings it is significantly better than greedy in 9 and never significantly worse, with gains in large-cascade regimes; on four empirical and synthetic networks, in 5 of 40 settings, with small typical differences. Integer programs over sampled scenarios did not beat greedy or DEFER and ran orders of magnitude slower."""
words = len(re.sub(r"\\[A-Za-z]+", "", ABSTRACT).split()); print("abstract words:", words); assert words <= 250
KEYWORDS = ["influence minimization", "adaptive optimization", "misinformation blocking", "adaptivity gap", "network optimization"]

# --- citation order: main text, then appendices, then end matter
END_MATTER = r"""
\section*{CRediT authorship contribution statement}
\textbf{Quang-Vinh Dang:} Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing -- original draft, Writing -- review \& editing. \textbf{Dat Le:} [roles to be confirmed by the author]. \textbf{Minh Ngoc Dinh:} [roles to be confirmed by the author].

\section*{Declaration of competing interest}
The authors declare that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section*{Funding}
This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

\section*{Data availability}
The code, scripts, networks used and per-instance results that reproduce every table and figure are available at \url{https://github.com/vinhqdang/trace_e}.

\section*{Declaration of generative AI and AI-assisted technologies in the manuscript preparation process}
During the preparation of this work the authors used Claude (Anthropic) to support the writing of the manuscript and to assist in writing and running the code, figures and diagrams. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.
"""
main = re.sub(r'\\paragraph\{([^}]*?)\.\}', r'\\paragraph{\1}', main); appendix = re.sub(r'\\paragraph\{([^}]*?)\.\}', r'\\paragraph{\1}', appendix)
order_text = main + appendix + END_MATTER
seen = []
for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", order_text):
    for k in m.group(1).split(","):
        k = k.strip()
        if k and k not in seen: seen.append(k)
missing = [k for k in seen if k not in bibs]; assert not missing, missing
unused = [k for k in bibs if k not in seen]; print("cited", len(seen), "unused bib entries:", unused)

def label_fix(k, lab):
    return lab.replace("Xie et~al.(2024)", "Xie et~al.(2025)")
bib_out = "\\begin{thebibliography}{99}\n\\providecommand{\\natexlab}[1]{#1}\n\\providecommand{\\url}[1]{\\texttt{#1}}\n\n"
for k in seen:
    lab, txt = bibs[k]
    bib_out += f"\\bibitem[{{{label_fix(k, lab)}}}]{{{k}}}\n{txt}\n\n"
bib_out += "\\end{thebibliography}\n"

PRE = r"""\documentclass[preprint,11pt]{elsarticle}
\usepackage{amsmath,amssymb,amsthm,amsfonts,bm}
\usepackage{graphicx}
\usepackage{array}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{tikz}
\usepackage{subcaption}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,calc,decorations.pathreplacing}
\usepackage[hidelinks]{hyperref}
\usepackage{setspace}
\emergencystretch=3em

\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newtheorem{proposition}{Proposition}
\newtheorem{corollary}{Corollary}
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\theoremstyle{remark}
\newtheorem{remark}{Remark}
% restated statements in the appendix
\newenvironment{repeattheorem}[1][]{\par\medskip\noindent\textbf{#1.}\ \itshape}{\par\medskip}
\newenvironment{repeatlemma}[1][]{\par\medskip\noindent\textbf{#1.}\ \itshape}{\par\medskip}
\newenvironment{repeatproposition}[1][]{\par\medskip\noindent\textbf{#1.}\ \itshape}{\par\medskip}
\newenvironment{repeatremark}[1][]{\par\medskip\noindent\textbf{#1.}\ \itshape}{\par\medskip}
\newcommand{\up}{\rule{0pt}{2.6ex}}
\newcommand{\down}{\rule[-1.2ex]{0pt}{0pt}}
\newcommand{\Rspread}{R}
\newcommand{\FF}{F}

\journal{Operations Research Perspectives}

\begin{document}

\begin{frontmatter}
\title{""" + title + r"""}
\author[a]{Quang-Vinh Dang}
\ead{vinh.dq4@buv.edu.vn}
\author[b]{Dat Le}
\ead{datla@uef.edu.vn}
\author[c]{Minh Ngoc Dinh\corref{cor1}}
\ead{minh.dinh@maeducation.com}
\cortext[cor1]{Corresponding author.}
\affiliation[a]{organization={British University Vietnam},
            city={Hung Yen},
            country={Vietnam}}
\affiliation[b]{organization={Department of Science, Technology, and International Projects, Ho Chi Minh City University of Economics and Finance},
            city={Ho Chi Minh City},
            postcode={700000},
            country={Vietnam}}
\affiliation[c]{organization={Millennia Education},
            city={Ho Chi Minh City},
            country={Vietnam}}

\begin{abstract}
""" + ABSTRACT + r"""
\end{abstract}

\begin{keyword}
""" + " \\sep ".join(KEYWORDS) + r"""
\end{keyword}
\end{frontmatter}

"""
APPX = "\n\\appendix\n\\counterwithin{table}{section}\n\\counterwithin{figure}{section}\n\\counterwithin{equation}{section}\n\\renewcommand{\\thetable}{\\Alph{section}.\\arabic{table}}\n\\renewcommand{\\thefigure}{\\Alph{section}.\\arabic{figure}}\n\\renewcommand{\\theequation}{\\Alph{section}.\\arabic{equation}}\n" + appendix
doc = PRE + main + APPX + END_MATTER + "\n" + bib_out + "\n\\end{document}\n"
os.makedirs(OUT, exist_ok=True)
shutil.copytree('/home/user/trace_e/paper/defer/figures', OUT + '/figures', dirs_exist_ok=True)
open(OUT + '/defer_orp.tex', 'w').write(doc)
print("written", len(doc))
