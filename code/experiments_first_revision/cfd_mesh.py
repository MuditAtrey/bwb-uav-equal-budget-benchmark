"""Half-model inviscid CFD mesh (Gmsh tets): wing + winglet from the pipeline geometry, symmetry plane at y=0.
(Gmsh 4.15 has no 3D boundary-layer field, so viscous effects are handled by 2D RANS sections, see cfd_airfoil.py.)
usage: python3 cfd_mesh.py DESIGN_ID OUTDIR [h_wall_mm=6] [first_layer_mm=0.35] [nlayers=14] [growth=1.25]"""
import sys, os, json, time, numpy as np, pandas as pd, gmsh
sys.path.insert(0, os.path.dirname(__file__))
from cfd_geom import sections, KEYS

def af_resample(P, w, xg, zg, le, chord, n=70, tte=0.0006):
    """Return 3D points around the airfoil (TE upper -> LE -> TE lower) with a small blunt TE, from the 3D section points P."""
    af = np.stack([((P - le) @ xg) / chord, ((P - le) @ zg) / chord], axis=1)
    iLE = int(np.argmin(af[:, 0]))
    up = af[:iLE + 1][::-1]      # LE -> TE upper (assumes coordinates start at TE upper)
    lo = af[iLE:]                # LE -> TE lower
    if up[:, 1].mean() < lo[:, 1].mean():
        up, lo = lo[::-1][::-1], up
    xs = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    yu = np.interp(xs, up[:, 0], up[:, 1]); yl = np.interp(xs, lo[:, 0], lo[:, 1])
    fade = np.clip((xs - 0.9) / 0.1, 0, 1)
    yu = yu + 0.5 * tte / chord * fade; yl = yl - 0.5 * tte / chord * fade
    A = np.concatenate([np.stack([xs[::-1], yu[::-1]], 1), np.stack([xs[1:], yl[1:]], 1)])  # TE upper -> LE -> TE lower
    return le + chord * (A[:, [0]] * xg + A[:, [1]] * zg)

def build(design_vec, out, h_wall=6.0, first=0.35, nlay=14, growth=1.25, h_far=300.0, alpha_deg=None):
    ap, w, S = sections(design_vec)
    os.makedirs(out, exist_ok=True)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("wing")
    O = gmsh.model.occ
    wires = []
    for i, xs in enumerate(w.xsecs):
        if i == 0:   # extra section below the symmetry plane so the boolean cut leaves a clean hole at y=0
            xg, yg, zg = w._compute_frame_of_WingXSec(0)
            pts = af_resample(S[0], w, xg, zg, np.array(xs.xyz_le, float), float(xs.chord)) - np.array([0, 0.03, 0])
            pt = [O.addPoint(*p) for p in pts]; sp = O.addSpline(pt); ln = O.addLine(pt[-1], pt[0]); wires.append(O.addWire([sp, ln]))
        xg, yg, zg = w._compute_frame_of_WingXSec(i)
        pts = af_resample(S[i], w, xg, zg, np.array(xs.xyz_le, float), float(xs.chord))
        ptags = [O.addPoint(*p) for p in pts]
        sp = O.addSpline(ptags)
        ln = O.addLine(ptags[-1], ptags[0])
        wires.append(O.addWire([sp, ln]))
    wing = O.addThruSections(wires, makeSolid=True, makeRuled=True)
    O.synchronize()
    X0, X1, Y1, Z0, Z1 = -1.5, 3.5, 3.0, -2.0, 2.0
    box = O.addBox(X0, 0.0, Z0, X1 - X0, Y1, Z1 - Z0)
    fluid = O.cut([(3, box)], wing)
    O.synchronize()
    wing_s, far_s, sym_s = [], [], []
    for d, t in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(d, t)
        on = lambda v, c: abs(v - c) < 1e-4
        if on(bb[1], 0) and on(bb[4], 0): sym_s.append(t)
        elif on(bb[0], X0) and on(bb[3], X0) or on(bb[0], X1) and on(bb[3], X1) or on(bb[1], Y1) and on(bb[4], Y1) or on(bb[2], Z0) and on(bb[5], Z0) or on(bb[2], Z1) and on(bb[5], Z1): far_s.append(t)
        else: wing_s.append(t)
    gmsh.model.addPhysicalGroup(2, wing_s, 1, name="wing"); gmsh.model.addPhysicalGroup(2, far_s, 2, name="farfield")
    gmsh.model.addPhysicalGroup(2, sym_s, 3, name="symmetry"); gmsh.model.addPhysicalGroup(3, [fluid[0][0][1]], 4, name="fluid")
    print("surfaces wing/far/sym:", len(wing_s), len(far_s), len(sym_s), flush=True)
    F = gmsh.model.mesh.field
    # size fields
    d = F.add("Distance"); F.setNumbers(d, "SurfacesList", wing_s); F.setNumber(d, "Sampling", 40)
    th = F.add("Threshold"); F.setNumber(th, "InField", d); F.setNumber(th, "SizeMin", h_wall * 1e-3); F.setNumber(th, "SizeMax", h_far * 1e-3)
    F.setNumber(th, "DistMin", 0.03); F.setNumber(th, "DistMax", 1.2)
    # wake box downstream of the wing
    bx = F.add("Box"); F.setNumber(bx, "VIn", 25e-3); F.setNumber(bx, "VOut", h_far * 1e-3)
    F.setNumber(bx, "XMin", 0.0); F.setNumber(bx, "XMax", 1.6); F.setNumber(bx, "YMin", 0.0); F.setNumber(bx, "YMax", 0.95)
    F.setNumber(bx, "ZMin", -0.25); F.setNumber(bx, "ZMax", 0.30); F.setNumber(bx, "Thickness", 0.5)
    mn = F.add("Min"); F.setNumbers(mn, "FieldsList", [th, bx]); F.setAsBackgroundMesh(mn)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0); gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24); gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0004)
    gmsh.option.setNumber("Mesh.Algorithm", 6); gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("General.NumThreads", 2)
    t0 = time.time(); gmsh.model.mesh.generate(2); print("2D done", round(time.time() - t0), "s", flush=True)
    gmsh.model.mesh.generate(3); print("3D done", round(time.time() - t0), "s", flush=True)
    nn = len(gmsh.model.mesh.getNodes()[0]); print("nodes", nn, flush=True)
    gmsh.option.setNumber("Mesh.Format", 42)   # SU2 .su2 format
    gmsh.write(os.path.join(out, "mesh.su2"))
    gmsh.finalize()
    return ap, w, nn

if __name__ == "__main__":
    al = pd.read_csv("/mnt/user-data/outputs/results_revision/al/al_seed0.csv")
    b = al[al.is_valid].nlargest(1, "range_calm_km").iloc[0]
    build(b[KEYS].values, sys.argv[1] if len(sys.argv) > 1 else "/tmp/cfdtest", h_wall=float(sys.argv[2]) if len(sys.argv) > 2 else 12.0)
