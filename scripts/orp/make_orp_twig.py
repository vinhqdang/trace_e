import re, sys, os, shutil
sys.path.insert(0, '.')
import orpbib
SRC = '/home/user/trace_e/paper/twig/twig.tex'
OUT = '/home/user/trace_e/paper/orp-twig'
t = open(SRC).read()

title = re.search(r"\\title\{(.*?)\}", t).group(1)
body_i = t.index("\\section{Introduction}"); bib_i = t.index("\\begin{thebibliography}"); bib_j = t.index("\\end{thebibliography}") + len("\\end{thebibliography}")
app_i = t.index("\\appendix"); end_i = t.index("\\end{document}")
main = t[body_i:bib_i].rstrip() + "\n"; appendix = t[app_i + len("\\appendix"):end_i].strip() + "\n"
bibs = orpbib.build(t)

ABSTRACT = re.search(r"\\begin\{abstract\}\n(.*?)\\end\{abstract\}", t, re.S).group(1).strip()
words = len(re.sub(r"\\[A-Za-z]+", "", ABSTRACT).split()); print("abstract words:", words); assert words <= 250
KEYWORDS = ["influence minimization", "exact computation", "pathwidth", "dynamic programming", "network reliability"]

# --- citation order: main text, then appendices, then end matter
END_MATTER = r"""
\section*{CRediT authorship contribution statement}
\textbf{Quang-Vinh Dang:} Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing -- original draft, Writing -- review \& editing. \textbf{Dat Le:} Writing -- review \& editing. \textbf{Minh Ngoc Dinh:} Writing -- review \& editing.

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
for a_, b_ in (('centre', 'center'), ('Centre', 'Center')):
    main = main.replace(a_, b_); appendix = appendix.replace(a_, b_)
main = re.sub(r'\\textbf\{Code and data availability\.\}[^\n]*\n', '', main)
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
shutil.copytree('/home/user/trace_e/paper/twig/figures', OUT + '/figures', dirs_exist_ok=True)
open(OUT + '/twig_orp.tex', 'w').write(doc)
print("written", len(doc))
