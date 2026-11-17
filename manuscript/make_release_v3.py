"""Builds the Zenodo upload for the current manuscript (2026-11-17). Unlike make_release.py (v2), nothing is withheld: run histories keep the
full design vectors, the evaluation database is included, and so is the evaluator code the study used. Output (default ../zenodo_v3/):
  Atrey_EqualBudget_BWB_preprint.pdf, Atrey_EqualBudget_BWB_supplement.pdf   manuscript and supplementary material
  data_and_code.zip       run histories, results, evaluator, experiment scripts, manuscript sources and generators
  cfd_solution_files.zip  SU2 restart (solution) files of all Euler cases
usage: python3 make_release_v3.py [OUTDIR]"""
import glob, os, shutil, sys, zipfile
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(f"{HERE}/../../../..")
OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/../zenodo_v3"); STAGE = f"{OUT}/_stage"
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(STAGE)
R, V3 = f"{REPO}/revision_runs/results_revision", f"{REPO}/revision_runs/results_v3"
skip = shutil.ignore_patterns("__pycache__", "*.pyc", "*_restart.dat", ".DS_Store", "paused_pids.txt")
def cp(src, dst):
    d = f"{STAGE}/{dst}"; os.makedirs(os.path.dirname(d), exist_ok=True)
    (shutil.copytree(src, d, ignore=skip, dirs_exist_ok=True) if os.path.isdir(src) else shutil.copy2(src, d))
# data
cp(f"{REPO}/flying_wing_optimizer/results/database.csv", "data/original_campaign/database.csv")
for x in sorted(os.listdir(R)):
    if x not in ("scripts", "figs", "analysis"): cp(f"{R}/{x}", f"data/results_revision/{x}")
for x in sorted(os.listdir(V3)):
    if x == "scripts" or x.endswith(".sh"): continue
    cp(f"{V3}/{x}", f"data/results_v3/{x}")
# code
for m in ["__init__", "main", "config", "geometry", "aerodynamics", "mission", "database", "surrogate", "optimizer", "visualization"]:
    p = f"{REPO}/flying_wing_optimizer/{m}.py"
    if os.path.exists(p): cp(p, f"code/flying_wing_optimizer/{m}.py")
cp(f"{R}/scripts", "code/experiments_first_revision"); cp(f"{R}/analysis", "code/experiments_first_revision/analysis")
cp(f"{V3}/scripts", "code/experiments"); [cp(f, f"code/experiments/launchers/{os.path.basename(f)}") for f in sorted(glob.glob(f"{V3}/run_*.sh"))]
# manuscript sources and generators
for f in ["main_revised.tex", "supplement.tex", "supplement_body.tex", "abstract_revised.tex", "references.bib", "elsarticle-num.bst", "figstyle.py", "response_to_reviewers.tex"] + sorted(glob.glob(f"{HERE}/numbers*.tex")) + sorted(glob.glob(f"{HERE}/outcome_macros.tex")) + sorted(glob.glob(f"{HERE}/make_*.py")):
    f = f if os.path.isabs(f) else f"{HERE}/{f}"
    if os.path.exists(f) and "make_release.py" not in f and "make_split.py" not in f: cp(f, f"manuscript/{os.path.basename(f)}")
cp(f"{HERE}/sections_revised", "manuscript/sections_revised"); cp(f"{HERE}/tables", "manuscript/tables")
for f in sorted(glob.glob(f"{HERE}/figures/*.pdf")) + sorted(glob.glob(f"{HERE}/figures/svg/*.svg")) + sorted(glob.glob(f"{HERE}/figures/*.png")):
    cp(f, "manuscript/figures/" + os.path.relpath(f, f"{HERE}/figures"))
open(f"{STAGE}/README.md", "w").write(open(f"{HERE}/ZENODO_V3_README.md").read())
shutil.copy2(f"{HERE}/preprint.pdf", f"{OUT}/Atrey_EqualBudget_BWB_preprint.pdf")   # built by make_preprint.py
shutil.copy2(f"{HERE}/supplement.pdf", f"{OUT}/Atrey_EqualBudget_BWB_supplement.pdf")
def zipdir(zpath, root, filt):
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for d, _, fs in os.walk(root):
            for f in sorted(fs):
                p = os.path.join(d, f)
                if filt(p): z.write(p, os.path.relpath(p, root)); n += 1
    return n
n1 = zipdir(f"{OUT}/data_and_code.zip", STAGE, lambda p: True)
os.makedirs(f"{OUT}/_cfd", exist_ok=True); n2 = 0
with zipfile.ZipFile(f"{OUT}/cfd_solution_files.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for base, tag in [(f"{R}/cfd", "results_revision/cfd"), (f"{V3}/cfd", "results_v3/cfd"), (f"{V3}/cfd_oos", "results_v3/cfd_oos")]:
        for p in sorted(glob.glob(f"{base}/**/*_restart.dat", recursive=True)): z.write(p, f"{tag}/{os.path.relpath(p, base)}"); n2 += 1
shutil.rmtree(STAGE); shutil.rmtree(f"{OUT}/_cfd")
for f in sorted(os.listdir(OUT)): print(f"{os.path.getsize(f'{OUT}/{f}') / 1e6:8.1f} MB  {f}")
print("files in data_and_code.zip:", n1, " CFD solution files:", n2)
