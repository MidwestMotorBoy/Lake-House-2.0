import numpy as np, sys, pickle, zipfile, io, contextlib, os, re
sys.path.insert(0, sys.argv[3])
from meshlib import mesh_bool, write_stl
W, OUT = sys.argv[1], sys.argv[2]
D = pickle.load(open(f'{W}/d_meta.pkl', 'rb')); zs = D['zs']; NY = D['NY']
G = 0.1; SC = 75.0; S = 304.8 / SC
X0, Y0 = 5.5, 3.0
COL = {1: ('Ground', '#7F9160'), 2: ('Stone and interior', '#D2C8B4'), 3: ('Siding and roofs', '#4B3A2D'), 4: ('Glass', '#8DB6D0')}
def obj_xml(oid, name, pidx, Pm):
    V = np.round(Pm.reshape(-1, 3), 4); U, inv = np.unique(V, axis=0, return_inverse=True); T = inv.reshape(-1, 3)
    return ''.join([f'<object id="{oid}" type="model"><mesh><vertices>',
        ''.join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in U), '</vertices><triangles>',
        ''.join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in T), '</triangles></mesh></object>'])
CT = '<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/><Default Extension="config" ContentType="application/xml"/></Types>'
RELS = '<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>'
PROJ = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'project_settings.config')).read() if '__file__' in globals() else open('code/project_settings.config').read()
def write_3mf(path, title, parts):
    """Bambu Studio flavoured 3MF: one object, one part per colour, part -> filament slot in Metadata/model_settings.config."""
    n = len(parts); top = n + 1
    m = ['<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:BambuStudio="http://schemas.bambulab.com/package/2021">',
         '<metadata name="Application">BambuStudio-02.08.02.61</metadata><metadata name="BambuStudio:3mfVersion">1</metadata>',
         f'<metadata name="Title">{title}</metadata><resources>']
    for i, (L, Pm) in enumerate(parts): m.append(obj_xml(i + 1, COL[L][0], i, Pm))
    m.append(f'<object id="{top}" type="model"><components>' + ''.join(f'<component objectid="{i + 1}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>' for i in range(n)) + '</components></object>')
    allv = np.concatenate([Pm.reshape(-1, 3) for L, Pm in parts]); lo = allv.min(0); hi = allv.max(0)
    tx, ty, tz = 165 - (lo[0] + hi[0]) / 2, 160 - (lo[1] + hi[1]) / 2, -lo[2]
    m.append(f'</resources><build><item objectid="{top}" transform="1 0 0 0 1 0 0 0 1 {tx:.4f} {ty:.4f} {tz:.4f}" printable="1"/></build></model>')
    c = ['<?xml version="1.0" encoding="UTF-8"?>\n<config>\n', f'  <object id="{top}">\n    <metadata key="name" value="{title}"/>\n    <metadata key="extruder" value="{parts[0][0]}"/>\n']
    for i, (L, Pm) in enumerate(parts):
        c.append(f'    <part id="{i + 1}" subtype="normal_part">\n      <metadata key="name" value="{L} {COL[L][0]}"/>\n      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n      <metadata key="extruder" value="{L}"/>\n    </part>\n')
    c.append('  </object>\n</config>\n')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', CT); z.writestr('_rels/.rels', RELS); z.writestr('3D/3dmodel.model', ''.join(m))
        z.writestr('Metadata/model_settings.config', ''.join(c))
        z.writestr('Metadata/project_settings.config', PROJ)
xi = lambda x: int(round((x - X0) / G)); yi = lambda y: int(round((y - Y0) / G))
jobs = []      # (filename stem, tier, lab, k0, j0, i0)
for n, me in enumerate(D['meta']):
    lab = np.load(f'{W}/d_{n:02d}.npy')
    if me['name'] == 'Base':
        J0, J1, I0, I1 = me['j0'], me['j1'], me['i0'], me['i1']
        ny, nx = J1 - J0, I1 - I0
        cx1, cx2, cy = xi(53.5) - I0, xi(102.5) - I0, yi(48.5) - J0
        jj, ii = np.mgrid[0:ny, 0:nx]
        T = (jj >= cy) * 3 + (ii >= cx1) * 1 + (ii >= cx2) * 1
        tabs = np.zeros((ny, nx), bool)
        DEP, NECK, HEAD = 16, 8, 13
        def tab_x(cx, yft):            # tab from the west tile into the east tile
            c = yi(yft) - J0; d = ii - cx
            m = (d >= 0) & (d < DEP) & (abs(jj - c) <= NECK + (HEAD - NECK) * d / DEP)
            T[m] = T[c, cx - 1]; tabs[m] = True
        def tab_y(xft):                # tab from the south tile into the north tile
            c = xi(xft) - I0; d = cy - 1 - jj
            m = (d >= 0) & (d < DEP) & (abs(ii - c) <= NECK + (HEAD - NECK) * d / DEP)
            T[m] = T[cy, c]; tabs[m] = True
        for yft in (10.0, 45.2, 70.0, 92.0): tab_x(cx1, yft)
        for yft in (10.0, 34.0, 70.0, 92.0): tab_x(cx2, yft)
        for xft in (12.0, 30.0, 66.0, 93.0, 116.0, 138.0): tab_y(xft)
        from scipy import ndimage as ndi
        gap = ndi.binary_dilation(tabs, structure=np.ones((3, 3), bool)) & ~tabs
        names = ['north-west', 'north-centre', 'north-east', 'south-west', 'south-centre', 'south-east']
        for t in range(6):
            m = (T == t) & ~gap
            ys = np.nonzero(m.any(axis=1))[0]; xs = np.nonzero(m.any(axis=0))[0]
            sub = lab[:, ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1] * m[None, ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
            jobs.append((f'A{t + 1} Base {names[t]}', 0, sub, me['k0'], J0 + ys[0], I0 + xs[0]))
    else:
        tier = 2 if 'roof' in me['name'] else 1
        jobs.append((me['name'], tier, lab, me['k0'], me['j0'], me['i0']))
lb = 0; lc = 0; summary = []
os.makedirs(f'{OUT}/print', exist_ok=True); os.makedirs(f'{OUT}/stl', exist_ok=True)
for d in ('asm', 'open', 'expl'): os.makedirs(f'{OUT}/r_{d}', exist_ok=True)
for (stem, tier, lab, k0, j0, i0) in jobs:
    if tier == 1: lb += 1; stem = f'B{lb} {stem}'
    if tier == 2: lc += 1; stem = f'C{lc} {stem}'
    nz = np.nonzero(lab.any(axis=(1, 2)))[0]; ny = np.nonzero(lab.any(axis=(0, 2)))[0]; nx = np.nonzero(lab.any(axis=(0, 1)))[0]
    a0, a1, b0, b1, c0, c1 = nz[0], nz[-1] + 1, ny[0], ny[-1] + 1, nx[0], nx[-1] + 1
    sub = lab[a0:a1, b0:b1, c0:c1]; zz = zs[k0 + a0:k0 + a1 + 1]
    off = np.array([(i0 + c0) * G * S, (NY - (j0 + b1)) * G * S, (zz[0] - zs[0]) * S])
    parts = []; tri = 0; bad = 0; vol = 0
    for L in (1, 2, 3, 4):
        m = sub == L
        if not m.any(): continue
        with contextlib.redirect_stdout(io.StringIO()):
            Pm, nrm, st = mesh_bool(m, zz, SC)
        parts.append((L, Pm)); tri += len(Pm); bad += st['open_edges']; vol += st['volume_cm3']
        write_stl(f'{OUT}/stl/{stem} - {L} {COL[L][0]}.stl', Pm, nrm, stem.encode())
        g = Pm + off
        for d, dz in (('asm', 0), ('expl', (0, 60, 120)[tier])):
            write_stl(f'{OUT}/r_{d}/{L}_{stem}.stl', g + np.array([0, 0, dz]), nrm)
        if tier < 2: write_stl(f'{OUT}/r_open/{L}_{stem}.stl', g, nrm)
    write_3mf(f'{OUT}/print/{stem}.3mf', stem, parts)
    dims = ((c1 - c0) * G * S, (b1 - b0) * G * S, (zz[-1] - zz[0]) * S)
    summary.append((stem, dims, len(parts), tri, bad, vol))
    print(f'{stem:44s} {dims[0]:5.0f} x {dims[1]:5.0f} x {dims[2]:5.0f} mm  colours {len(parts)}  tris {tri:7d}  open edges {bad}  {vol:6.0f} cm3', flush=True)
pickle.dump(summary, open(f'{OUT}/summary.pkl', 'wb'))
print('total volume cm3', round(sum(s[5] for s in summary)))
