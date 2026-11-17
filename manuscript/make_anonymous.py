"""Builds main_anonymous.tex (double-anonymized review copy) from main_revised.tex: removes the author block, the CRediT statement, the
acknowledgments and the repository URL, and rewords the one self-reference in Section 4. usage: python3 make_anonymous.py"""
m = open("main_revised.tex").read()
a = m.index(r"\author[kit]"); b = m.index(r"\begin{abstract}"); m = m[:a] + "% author block removed for double-anonymized review (see title page)\n\n" + m[b:]
a = m.index(r"\section*{CRediT Author Statement}"); b = m.index(r"\section*{Data and Code Availability}"); m = m[:a] + m[b:]
u = r"\url{https://doi.org/10.5281/zenodo.22893592}"; assert m.count(u) == 1; m = m.replace(u, "a public repository (link removed for double-anonymized review; given on the title page)")
a = m.index(r"\section*{Acknowledgments}"); b = m.index(r"\bibliographystyle"); m = m[:a] + m[b:]
k = r"\input{sections_revised/04_geometric_manifold_and_trim.tex}"; assert m.count(k) == 1; m = m.replace(k, r"\input{sections_anon_04.tex}")
open("main_anonymous.tex", "w").write(m)
s4 = open("sections_revised/04_geometric_manifold_and_trim.tex").read(); p = "the preprint of this work"; assert s4.count(p) == 1
open("sections_anon_04.tex", "w").write("% generated copy for the anonymized build; do not edit\n" + s4.replace(p, "an earlier version of this work"))
