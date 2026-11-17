"""Builds preprint.tex (public preprint for Zenodo) from main_revised.tex: no journal named in the footer (the venue is not decided and a
preprint should not imply acceptance), and a footer that marks it as a non-peer-reviewed preprint with its date and licence.
usage: python3 make_preprint.py ; then tectonic --keep-intermediates preprint.tex"""
import datetime
m = open("main_revised.tex").read()
j = [l for l in m.splitlines() if l.startswith("\\journal{")]; assert len(j) == 1
m = m.replace(j[0], "\\journal{}")
foot = (r"\makeatletter" "\n"
        r"\def\ps@pprintTitle{\let\@oddhead\@empty\let\@evenhead\@empty" "\n"
        r"  \def\@oddfoot{\footnotesize\itshape Preprint, not peer reviewed. Version of " + datetime.date.today().strftime("%-d %B %Y") + r". Licensed under CC BY 4.0.\hfill}\let\@evenfoot\@oddfoot}" "\n"
        r"\makeatother" "\n")
k = "\\begin{document}\n"; assert m.count(k) == 1
m = m.replace(k, foot + k)
open("preprint.tex", "w").write(m)
