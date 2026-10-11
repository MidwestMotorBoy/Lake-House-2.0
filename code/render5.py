"""Shaded orthographic renderer with cast shadows, ambient occlusion and anti-aliasing.
usage as a module: render(files, out, az, el, width=1600, title=None)   files = list of 'L_name.stl' (L = colour 1-4)"""
import numpy as np, math, os, glob, sys
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
DT = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
PAL = {'1': (122, 146, 92), '2': (214, 204, 184), '3': (86, 64, 48), '4': (132, 178, 208), 'g': (236, 234, 229)}

def raster(X, Y, Z, W, H, ids=None):
    zb = np.full((H, W), np.inf); ib = np.full((H, W), -1, np.int32)
    x0a = np.maximum(np.floor(X.min(1)).astype(int), 0); x1a = np.minimum(np.ceil(X.max(1)).astype(int) + 1, W)
    y0a = np.maximum(np.floor(Y.min(1)).astype(int), 0); y1a = np.minimum(np.ceil(Y.max(1)).astype(int) + 1, H)
    den = (Y[:, 1] - Y[:, 2]) * (X[:, 0] - X[:, 2]) + (X[:, 2] - X[:, 1]) * (Y[:, 0] - Y[:, 2])
    for t in np.nonzero((x1a > x0a) & (y1a > y0a) & (np.abs(den) > 1e-12))[0]:
        x0, x1, y0, y1 = x0a[t], x1a[t], y0a[t], y1a[t]; x = X[t]; y = Y[t]; z = Z[t]
        gx = np.arange(x0, x1) + 0.5; gy = (np.arange(y0, y1) + 0.5)[:, None]
        a = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / den[t]
        b = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / den[t]
        c = 1 - a - b; zz = a * z[0] + b * z[1] + c * z[2]
        sub = zb[y0:y1, x0:x1]; upd = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6) & (zz < sub - 1e-7)
        if upd.any():
            sub[upd] = zz[upd]
            if ids is not None: ib[y0:y1, x0:x1][upd] = ids[t]
    return zb, ib

def basis(az, el):
    az, el = math.radians(az), math.radians(el)
    f = np.array([-math.cos(el) * math.sin(az), -math.cos(el) * math.cos(az), -math.sin(el)])
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f); return r, u, f

def render(files, out, az, el, width=1600, title=None, ss=2, ground=True, laz=None, lel=52):
    Vs = []; Cs = []
    for p in files:
        d = np.fromfile(p, dtype=DT, offset=84); Vs.append(d['v'].astype(np.float64)); Cs.append(np.full(len(d), int(os.path.basename(p)[0])))
    V = np.concatenate(Vs); C = np.concatenate(Cs)
    lo = V.reshape(-1, 3).min(0); hi = V.reshape(-1, 3).max(0); span = (hi - lo).max()
    if ground:
        m = span * 1.5; z = lo[2] - 0.02
        q = np.array([[lo[0] - m, lo[1] - m, z], [hi[0] + m, lo[1] - m, z], [hi[0] + m, hi[1] + m, z], [lo[0] - m, hi[1] + m, z]])
        V = np.concatenate([V, q[[0, 1, 2]][None], q[[0, 2, 3]][None]]); C = np.concatenate([C, [0, 0]])
    r, u, f = basis(az, el); W = width * ss
    P = np.stack([V @ r, V @ u, V @ f], -1); M = P[C > 0].reshape(-1, 3)
    mn = M.min(0); mx = M.max(0); pad = 0.07 * (mx[0] - mn[0]); sc = W / (mx[0] - mn[0] + 2 * pad)
    H = int((mx[1] - mn[1] + 2 * pad) * sc) + (int(0.09 * W) if title else 0)
    X = (P[..., 0] - mn[0] + pad) * sc; Y = (mx[1] - P[..., 1] + pad) * sc
    nrm = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]); nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)[:, None]
    front = nrm @ (-f) > 1e-6
    zb, ib = raster(X[front], Y[front], P[front][..., 2], W, H, np.nonzero(front)[0])
    hit = ib >= 0
    # light and shadow map
    lr, lu, lf = basis(az - 55 if laz is None else laz, lel); L = -lf
    Q = np.stack([V @ lr, V @ lu, V @ lf], -1); solid = C > 0; qm = Q[solid].reshape(-1, 3); qn = qm.min(0); qx = qm.max(0)
    SW = 3072; ssc = (SW - 8) / max(qx[0] - qn[0], qx[1] - qn[1])
    sm, _ = raster((Q[solid][..., 0] - qn[0]) * ssc + 4, (Q[solid][..., 1] - qn[1]) * ssc + 4, Q[solid][..., 2], SW, SW)
    yy, xx = np.nonzero(hit)
    wp = np.outer((xx + 0.5) / sc + mn[0] - pad, r) + np.outer(mx[1] + pad - (yy + 0.5) / sc, u) + np.outer(zb[yy, xx], f)
    sx = np.clip(((wp @ lr - qn[0]) * ssc + 4).astype(int), 0, SW - 1); sy = np.clip(((wp @ lu - qn[1]) * ssc + 4).astype(int), 0, SW - 1)
    lit = np.ones(len(yy))
    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        lit += (wp @ lf <= sm[np.clip(sy + dy, 0, SW - 1), np.clip(sx + dx, 0, SW - 1)] + span * 0.004)
    lit = (lit - 1) / 5.0
    n = nrm[ib[yy, xx]]; ndl = np.clip(n @ L, 0, 1); cid = C[ib[yy, xx]]
    lit = lit * (ndl > 0.02)
    # ambient occlusion from the depth buffer
    zf = np.where(hit, zb, np.nan); far = np.nanmax(zf); zf = np.where(hit, zb, far)
    ao = np.zeros((H, W))
    for s in (4 * ss, 12 * ss): ao += np.clip((zf - ndi.gaussian_filter(zf, s)) / (span * 0.012), 0, 1)
    ao = 1 - 0.38 * np.clip(ao / 2, 0, 1)
    pal = np.array([PAL['g'], PAL['1'], PAL['2'], PAL['3'], PAL['4']], float)
    base = pal[cid]; amb = 0.50 + 0.14 * n[:, 2]
    shade = amb + 0.58 * ndl * lit
    spec = (cid == 4) * 0.25 * np.clip((n @ ((L - f) / np.linalg.norm(L - f))), 0, 1) ** 20 * lit
    rgb = np.empty((H, W, 3)); g = np.linspace(244, 226, H)[:, None, None]; rgb[:] = g * np.array([1.0, 0.995, 0.985])
    px = base * (shade * ao[yy, xx])[:, None] + 255 * spec[:, None]
    isg = cid == 0
    if ground:                                     # ground only shows as a soft shadow on the backdrop
        px[isg] = rgb[yy[isg], xx[isg]] * (0.66 + 0.34 * lit[isg])[:, None] * (0.85 + 0.15 * ao[yy[isg], xx[isg]])[:, None]
    rgb[yy, xx] = px
    # light outline at depth steps
    e = np.abs(np.diff(zf, axis=0, prepend=zf[:1])) + np.abs(np.diff(zf, axis=1, prepend=zf[:, :1]))
    edge = ndi.gaussian_filter(np.clip(e / (span * 0.004), 0, 1), 0.6 * ss) * (ib >= 0) * (C[np.clip(ib, 0, None)] > 0)
    rgb *= (1 - 0.30 * edge)[..., None]
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((W // ss, H // ss), Image.LANCZOS)
    if title:
        d = ImageDraw.Draw(img); fo = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', int(width * 0.026))
        d.text((int(width * 0.035), img.height - int(width * 0.062)), title, font=fo, fill=(60, 52, 46))
    img.save(out); return img.size
if __name__ == '__main__':
    print(render(sorted(glob.glob(sys.argv[1] + '/*.stl')), sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 1600, sys.argv[6] if len(sys.argv) > 6 else None))
