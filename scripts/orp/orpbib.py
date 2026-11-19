import re, json

JOURNALS = {
 "Management Sci.": "Manag Sci", "Science": "Science", "Nature": "Nature", "Phys.\\ Rev.\\ Lett.": "Phys Rev Lett", "Phys.\\ Rev.\\ E": "Phys Rev E",
 "Oper.\\ Res.": "Oper Res", "Math.\\ Oper.\\ Res.": "Math Oper Res", "J.\\ Artificial Intelligence Res.": "J Artif Intell Res", "SIAM J.\\ Optim.": "SIAM J Optim",
 "ACM Trans.\\ Programming Languages and Systems": "ACM Trans Program Lang Syst", "ACM Trans.\\ Knowledge Discovery from Data": "ACM Trans Knowl Discov Data",
 "Naval Res.\\ Logist.": "Nav Res Logist", "IEEE Trans.\\ Knowledge and Data Engrg.": "IEEE Trans Knowl Data Eng", "Automatica": "Automatica",
 "IEEE Trans.\\ Control of Network Systems": "IEEE Trans Control Netw Syst", "Oper.\\ Res.\\ Perspect.": "Oper Res Perspect", "Eur.\\ J.\\ Oper.\\ Res.": "Eur J Oper Res",
 "Comput.\\ Oper.\\ Res.": "Comput Oper Res", "IEEE/ACM Trans.\\ Networking": "IEEE/ACM Trans Netw", "IEEE Trans.\\ Comput.\\ Social Systems": "IEEE Trans Comput Soc Syst",
 "Proc.\\ VLDB Endowment": "Proc VLDB Endow", "World Wide Web": "World Wide Web", "INFORMS J.\\ Comput.": "INFORMS J Comput",
 "IEEE Trans.\\ Network Science and Engineering": "IEEE Trans Netw Sci Eng", "J.\\ Network and Computer Applications": "J Netw Comput Appl",
 "SIAM J.\\ Comput.": "SIAM J Comput", "Inform.\\ Process.\\ Lett.": "Inf Process Lett", "J.\\ Algorithms": "J Algorithms", "IEEE Trans.\\ Reliability": "IEEE Trans Reliab",
 "IEICE Trans.\\ Fundamentals of Electronics, Communications and Computer Sciences": "IEICE Trans Fundam Electron Commun Comput Sci",
}
EXPAND = [("Proc.\\ ", "Proceedings of the "), ("Internat.\\ ", "International "), ("Conf.\\ ", "Conference "), ("Sympos.\\ ", "Symposium "), ("Annual ", "Annual "), ("Sci.\\ ", "Science "), ("Web Sci.\\ ", "Web Science ")]
DOIS = {
 "asadpournazerzadeh2016": "10.1287/mnsc.2015.2254", "barabasi1999": "10.1126/science.286.5439.509", "borgs2014": "10.1137/1.9781611973402.70", "deangoemansvondrak2008": "10.1287/moor.1080.0330",
 "fan2013": "10.1109/ICDCS.2013.34", "gupta2016": "10.1137/1.9781611974331.ch120", "he2012": "10.1137/1.9781611972825.40", "kempe2003": "10.1145/956750.956769", "khalil2014": "10.1145/2623330.2623704",
 "kleywegt2002": "10.1137/S1052623499363220", "kuhlman2013": "10.1109/ICDM.2013.47", "lengauer1979": "10.1145/357062.357071", "leskovec2007b": "10.1145/1281192.1281239", "li2024": "10.1002/nav.22134",
 "li2018": "10.1109/TKDE.2018.2807843", "nguyen2012": "10.1145/2380718.2380746", "pastor2002": "10.1103/PhysRevE.65.036104", "rempel2021": "10.1016/j.orp.2021.100204", "roohnavazfar2019": "10.1016/j.orp.2019.100124",
 "seemansinger2013": "10.1109/FOCS.2013.56", "shi2019": "10.1109/ICDE.2019.00135", "smith2020": "10.1016/j.ejor.2019.06.024", "sun2018": "10.1145/3219819.3220101", "taninmis2022": "10.1016/j.ejor.2021.05.008",
 "taninmis2024": "10.1016/j.cor.2024.106675", "tong2012": "10.1145/2396761.2396795", "wang2016drimux": "10.1609/aaai.v30i1.10063", "wang2024sandwich": "10.14778/3675034.3675042", "xie2023": "10.1109/ICDE55515.2023.00066",
 "xie2024": "10.1287/ijoc.2024.0591", "yuan2017": "10.24963/ijcai.2017/546", "chen2010": "10.1145/1835804.1835934", "cohen2003": "10.1103/PhysRevLett.91.247901", "cormican1998": "10.1287/opre.46.2.184",
 "mayne2000": "10.1016/S0005-1098(99)00214-9", "preciado2014": "10.1109/TCNS.2014.2310911", "tang2014": "10.1145/2588555.2593670", "yan2020": "10.1109/TNSE.2019.2903272", "zareie2021": "10.1016/j.jnca.2021.103094",
  "bodlaender1988": "10.1007/3-540-19488-6_110", "bodlaender1996": "10.1006/jagm.1996.0049", "budak2011": "10.1145/1963405.1963499", "hardy2007": "10.1109/TR.2007.898572",
 "kinnersley1992": "10.1016/0020-0190(92)90234-M", "maehara2017": "10.1145/3038912.3052567", "provan1983": "10.1137/0212053", "valiant1979": "10.1137/0208032",
 "kawahara2017": "10.1587/transfun.E100.A.1773", "satyanarayana1985": "10.1137/0214057",
 "birge2011": "10.1007/978-1-4614-0237-4", "cancela2021": "10.1016/j.orp.2021.100199",
}
OVERRIDE = {
 "xie2024": ("Xie J, Zhang F, Wang K, Liu J, Lin X, Zhang W", "Influence minimization via blocking strategies", "INFORMS J Comput 2025;37(6):1587-604"),
 "taninmis2022": ("Tanınmış K, Aras N, Altınel İK", "Improved $x$-space algorithm for min-max bilevel problems with an application to misinformation spread in social networks", "Eur J Oper Res 2022;297(1):40-52"),
 "taninmis2024": ("Tanınmış K, Aras N, Güney E, Sinnl M", "Benders decomposition algorithms for minimizing the spread of harmful contagions in networks", "Comput Oper Res 2024;167:106675"),
}

def short_pages(a, b):
    if len(a) == len(b):
        k = 0
        while k < len(a) - 1 and a[k] == b[k]: k += 1
        return f"{a}-{b[k:]}"
    return f"{a}-{b}"

def fmt_authors(a):
    parts = [x.strip() for x in a.split(",") if x.strip()]
    return ", ".join(parts[:6]) + (", et al" if len(parts) > 6 else "")

ACC = {"{\\'E}": "É", "\\'E": "É", "{\\'a}": "á", "\\'a": "á", "{\\'e}": "é", "\\'e": "é"}
def clean_tex(t):
    t = t.replace("\\ ", " ").replace("~", " ")
    for k, v in ACC.items(): t = t.replace(k, v)
    return t

def reformat(key, body):
    body = re.sub(r"\s+", " ", body.strip())
    if key in OVERRIDE:
        a, t, v = OVERRIDE[key]; doi = DOIS.get(key)
        return f"{a}. {t}. {v}." + (f" \\url{{https://doi.org/{doi}}}." if doi else "")
    if key == "birge2011":
        return "Birge JR, Louveaux F. Introduction to Stochastic Programming. 2nd ed. New York: Springer; 2011. \\url{https://doi.org/10.1007/978-1-4614-0237-4}."
    if key == "cygan2015":
        return "Cygan M, Fomin FV, Kowalik Ł, Lokshtanov D, Marx D, Pilipczuk M, et al. Parameterized Algorithms. Cham: Springer; 2015. \\url{https://doi.org/10.1007/978-3-319-21275-3}."
    if key == "nakamura2026":
        return "Nakamura K, Nishino M. Linear-time exact computation of influence spread on bounded-pathwidth graphs. In: 20th Scandinavian Symposium on Algorithm Theory (SWAT 2026), LIPIcs vol. 370, article 34; 2026."
    if key == "pozar2026":
        return "Požar M. Exact greedy influence maximization in linear time on bounded-treewidth graphs. Preprint, arXiv:2609.19960 [cs.DS]; 2026. \\url{https://arxiv.org/abs/2609.19960}."
    if key == "colbourn1987":
        return "Colbourn CJ. The Combinatorics of Network Reliability. New York: Oxford University Press; 1987."
    m = re.match(r"(.*?) \((\d{4})\) (.*?)\. (.*)$", body)
    if not m: return clean_tex(body)
    authors, year, title, rest = m.groups()
    authors = fmt_authors(clean_tex(authors)); title = clean_tex(title)
    doi = DOIS.get(key); tail = f" \\url{{https://doi.org/{doi}}}." if doi else ""
    em = re.match(r"\\emph\{(.*?)\}(.*)$", rest)
    venue, after = (em.group(1), em.group(2).strip()) if em else (rest, "")
    if venue in JOURNALS:
        v = JOURNALS[venue]; mm = re.match(r"(\d+)(\((\d+)\))?:(\d+)(?:--(\d+))?\.?$", after)
        if mm:
            vol, _, iss, p1, p2 = mm.groups(); loc = f"{vol}" + (f"({iss})" if iss else "") + ":" + (short_pages(p1, p2) if p2 else p1)
            return f"{authors}. {title}. {v} {year};{loc}.{tail}"
        mm = re.match(r"(\d+):(\d+)--(\d+)\.?$", after)
        if mm: return f"{authors}. {title}. {v} {year};{mm.group(1)}:{short_pages(mm.group(2), mm.group(3))}.{tail}"
        mm = re.match(r"(\d+)\(?(\d+)?\)?:Article (\d+)\.?$", after)
        if mm: return f"{authors}. {title}. {v} {year};{mm.group(1)}({mm.group(2)}):{mm.group(3)}.{tail}"
        return f"{authors}. {title}. {v} {year}{(';' + after.rstrip('.')) if after else ''}.{tail}"
    # proceedings
    v = venue
    for a_, b_ in EXPAND: v = v.replace(a_, b_)
    v = clean_tex(v).replace("Sympos.", "Symposium").replace("Sci.", "Science").replace("Engrg.", "Engineering")
    mm = re.match(r",?\s*(?:AAAI Technical Report WS-13-17, )?(\d+)--(\d+)\.?$", after)
    pp = f", p. {short_pages(mm.group(1), mm.group(2))}" if mm else ""
    return f"{authors}. {title}. In: {v}; {year}{pp}.{tail}"

def build(tex):
    i = tex.index("\\begin{thebibliography}"); j = tex.index("\\end{thebibliography}")
    bib = tex[i:j]; k0 = bib.index("\\bibitem")
    items = [x for x in re.split(r"(?=\\bibitem)", bib[k0:]) if x.strip()]
    out = {}
    for it in items:
        mkey = re.search(r"\]\{([^}]*)\}", it); key = mkey.group(1)
        label = re.search(r"\\bibitem\[\{(.*?)\}\]", it, re.S).group(1)
        body = it[it.index("}", it.index("]{")) + 1:]
        out[key] = (label, reformat(key, body))
    return out
