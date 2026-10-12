"""Perspective interior renderer for the voxel house model (feet, plan frame: x east, y south, z elevation)."""
import numpy as np, math, glob, pickle, os, sys
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
DT = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
S = 304.8 / 75

def load(out75, work):
    Vs, Ls = [], []
    for p in sorted(glob.glob(out75 + '/r_asm/*.stl')):
        d = np.fromfile(p, dtype=DT, offset=84)['v'].astype(np.float64) / S
        v = np.empty_like(d); v[..., 0] = d[..., 0] + 5.5; v[..., 1] = 97.0 - d[..., 1]; v[..., 2] = d[..., 2] + 93.5
        Vs.append(v); Ls.append(np.full(len(v), int(os.path.basename(p)[0])))
    V = np.concatenate(Vs); L = np.concatenate(Ls)
    D = pickle.load(open(work + '/d_meta.pkl', 'rb')); zs = D['zs']; NY, NX = D['NY'], D['NX']
    top = np.full((NY, NX), -1e9)                       # highest solid per column (for inside/outside)
    for n, m in enumerate(D['meta']):
        lab = np.load(f'{work}/d_{n:02d}.npy') > 0
        if not lab.any(): continue
        kk = np.where(lab.any(axis=(1, 2)))[0]
        hi = np.where(lab.any(0), (lab.shape[0] - 1 - np.argmax(lab[::-1], axis=0)), -1)
        zt = np.where(hi >= 0, zs[np.clip(m['k0'] + hi + 1, 0, len(zs) - 1)], -1e9)
        sl = top[m['j0']:m['j0'] + lab.shape[1], m['i0']:m['i0'] + lab.shape[2]]; np.maximum(sl, zt, out=sl)
    return V, L, top

def raster(X, Y, Z, W, H, ids):
    zb = np.full((H, W), np.inf); ib = np.full((H, W), -1, np.int64)
    x0a = np.maximum(np.floor(X.min(1)).astype(int), 0); x1a = np.minimum(np.ceil(X.max(1)).astype(int) + 1, W)
    y0a = np.maximum(np.floor(Y.min(1)).astype(int), 0); y1a = np.minimum(np.ceil(Y.max(1)).astype(int) + 1, H)
    den = (Y[:, 1] - Y[:, 2]) * (X[:, 0] - X[:, 2]) + (X[:, 2] - X[:, 1]) * (Y[:, 0] - Y[:, 2])
    for t in np.nonzero((x1a > x0a) & (y1a > y0a) & (np.abs(den) > 1e-12))[0]:
        x0, x1, y0, y1 = x0a[t], x1a[t], y0a[t], y1a[t]; x = X[t]; y = Y[t]; z = Z[t]
        gx = np.arange(x0, x1) + 0.5; gy = (np.arange(y0, y1) + 0.5)[:, None]
        a = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / den[t]
        b = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / den[t]
        c = 1 - a - b; zz = a * z[0] + b * z[1] + c * z[2]
        sub = zb[y0:y1, x0:x1]; upd = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6) & (zz < sub)
        if upd.any(): sub[upd] = zz[upd]; ib[y0:y1, x0:x1][upd] = ids[t]
    return zb, ib

def clip_near(C, ids, near):
    """C: (n,3,3) camera-space tris (z forward). Clip against z >= near; returns tris and ids."""
    inn = C[:, :, 2] >= near; k = inn.sum(1)
    out = [C[k == 3]]; oid = [ids[k == 3]]
    for t in np.nonzero((k == 1) | (k == 2))[0]:
        P = C[t]; poly = []
        for i in range(3):
            a, b = P[i], P[(i + 1) % 3]; ia, ibb = a[2] >= near, b[2] >= near
            if ia: poly.append(a)
            if ia != ibb: s = (near - a[2]) / (b[2] - a[2]); poly.append(a + s * (b - a))
        for j in range(1, len(poly) - 1): out.append(np.array([[poly[0], poly[j], poly[j + 1]]])); oid.append([ids[t]])
    return np.concatenate(out), np.concatenate([np.atleast_1d(o) for o in oid])

def hsh(*a):
    h = np.zeros_like(a[0], dtype=np.int64)
    for i, v in enumerate(a): h = h * 73856093 + (v.astype(np.int64) * (19349663 + 83492791 * i))
    h = (h ^ (h >> 13)) * 1274126177; return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0

def render(V, L, top, eye, look, out, fov=70, W=1600, H=1000, ss=2, title=None, sun=(235, 32), exposure=1.0, rugs=(), props=()):
    eye = np.array(eye, float); look = np.array(look, float)
    f = look - eye; f /= np.linalg.norm(f); r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    nrm = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]); ln = np.linalg.norm(nrm, axis=1); ok = ln > 1e-12
    nrm = -nrm / np.maximum(ln, 1e-12)[:, None]          # y flip on load reverses the winding
    # material per triangle
    cen = V.mean(1); ix = np.clip(((cen[:, 0] - 5.5) / 0.1).astype(int), 0, top.shape[1] - 1); jy = np.clip(((cen[:, 1] - 3.0) / 0.1).astype(int), 0, top.shape[0] - 1)
    inside = top[jy, ix] > cen[:, 2] + 2.0
    up = nrm[:, 2] > 0.7; dn = nrm[:, 2] < -0.7; vert = ~up & ~dn
    M = np.zeros(len(V), int)                       # material ids
    # 1 grass 2 stone-out 3 plaster 4 ceiling 5 oak floor 6 stone tile 7 walnut 8 fabric 9 dark siding 10 glass 11 black metal 12 paver
    M[L == 1] = 1
    m2 = L == 2
    M[m2 & ~inside] = 2; M[m2 & ~inside & up] = 12
    M[m2 & inside & vert] = 3; M[m2 & inside & dn] = 4
    M[m2 & inside & up] = 5; M[m2 & inside & up & (cen[:, 2] < 97.5)] = 6
    m3 = L == 3
    M[m3] = 7; M[m3 & ~inside] = 9
    fl = np.floor(cen[:, 2] * 4) / 4
    FL = np.array([94.0, 96.0, 98.5, 100.0, 107.0, 113.0, 114.0, 116.0])
    hgt = cen[:, 2] - FL[np.clip(np.searchsorted(FL, cen[:, 2] + 1e-3) - 1, 0, None)]
    ext = np.minimum(np.ptp(V[:, :, 0], axis=1), np.ptp(V[:, :, 1], axis=1))
    fur = m3 & inside & (hgt <= 3.4)
    M[fur & vert] = 14
    M[fur & up & (hgt <= 1.8)] = 8
    M[fur & up & (hgt > 2.1) & (ext >= 0.9)] = 7
    M[fur & up & (hgt > 2.85) & (ext >= 0.9)] = 13
    M[m2 & inside & up & (hgt > 0.3) & (hgt <= 3.4)] = 17          # bedding / cushions on label-2 tops
    for (x0, y0, x1, y1, z0, z1, mt, ms) in props:                  # named objects: top / side materials
        sel = (cen[:, 0] > x0) & (cen[:, 0] < x1) & (cen[:, 1] > y0) & (cen[:, 1] < y1) & (cen[:, 2] > z0) & (cen[:, 2] < z1) & (L != 4)
        M[sel & up] = mt; M[sel & ~up] = ms
    M[L == 4] = 10
    M[(L == 4) & up & (cen[:, 2] < 95.0)] = 15
    # camera pass (glass is see-through)
    vis = ok & (M != 10)
    C = np.stack([(V - eye) @ r, (V - eye) @ u, (V - eye) @ f], -1)[vis]
    C, cid = clip_near(C, np.nonzero(vis)[0], 0.3)
    fpx = (W * ss / 2) / math.tan(math.radians(fov / 2))
    X = W * ss / 2 + fpx * C[..., 0] / C[..., 2]; Y = H * ss / 2 - fpx * C[..., 1] / C[..., 2]
    zb, ib = raster(X, Y, -1.0 / C[..., 2], W * ss, H * ss, cid)
    hit = ib >= 0; yy, xx = np.nonzero(hit)
    dz = -1.0 / zb[yy, xx]
    dirc = np.stack([(xx + 0.5 - W * ss / 2) / fpx, -(yy + 0.5 - H * ss / 2) / fpx, np.ones(len(xx))], -1)
    wp = eye + (dirc * dz[:, None]) @ np.stack([r, u, f])
    t = ib[yy, xx]; n = nrm[t]; m = M[t]
    # flip normals toward the camera
    vdir = wp - eye; n = np.where(((n * vdir).sum(1) > 0)[:, None], -n, n)
    # sun + shadow map (glass does not cast)
    az, el = math.radians(sun[0]), math.radians(sun[1])
    Ld = np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])   # toward the sun, x east y north? (y south here)
    Ld[1] = -Ld[1]
    lf = -Ld; lr = np.cross(lf, [0, 0, 1.0]); lr /= np.linalg.norm(lr); lu = np.cross(lr, lf)
    sv = ok & (M != 10); Q = np.stack([V[sv] @ lr, V[sv] @ lu, V[sv] @ lf], -1)
    qn = Q.reshape(-1, 3).min(0); qx = Q.reshape(-1, 3).max(0); SW = 6000; ssc = (SW - 8) / max(qx[0] - qn[0], qx[1] - qn[1])
    sm, _ = raster((Q[..., 0] - qn[0]) * ssc + 4, (Q[..., 1] - qn[1]) * ssc + 4, Q[..., 2], SW, SW, np.zeros(len(Q), np.int64))
    sx = ((wp @ lr - qn[0]) * ssc + 4); sy = ((wp @ lu - qn[1]) * ssc + 4); lit = np.zeros(len(xx))
    for ox, oy in ((0, 0), (1.2, 0), (-1.2, 0), (0, 1.2), (0, -1.2), (0.8, 0.8), (-0.8, -0.8), (0.8, -0.8), (-0.8, 0.8)):
        lit += (wp @ lf <= sm[np.clip((sy + oy).astype(int), 0, SW - 1), np.clip((sx + ox).astype(int), 0, SW - 1)] + 0.12)
    lit /= 9; ndl = np.clip(n @ Ld, 0, 1) * lit
    # procedural material colours
    pal = {1: (104, 132, 78), 2: (176, 164, 146), 3: (232, 225, 214), 4: (238, 234, 227), 5: (176, 132, 88), 6: (196, 188, 174),
           7: (92, 64, 44), 8: (218, 208, 190), 9: (78, 58, 44), 11: (36, 36, 38), 12: (190, 182, 168), 13: (228, 224, 216), 14: (158, 144, 126), 15: (80, 150, 168), 16: (52, 104, 80), 17: (238, 233, 224), 18: (110, 72, 50), 19: (122, 86, 56)}
    base = np.array([pal.get(k, (200, 200, 200)) for k in range(20)], float)[m]
    x, y, z = wp[:, 0], wp[:, 1], wp[:, 2]
    s = m == 5                                        # oak planks along x
    if s.any():
        row = np.floor(y[s] / 0.55); off = hsh(row) * 7; plank = np.floor((x[s] + off) / 6.0)
        v = 0.86 + 0.22 * hsh(row, plank) + 0.05 * np.sin(x[s] * 9 + hsh(row) * 6)
        edge = (np.abs(((y[s] / 0.55) % 1) - 0.5) > 0.47) | (np.abs((((x[s] + off) / 6.0) % 1) - 0.5) > 0.495)
        base[s] *= (v * np.where(edge, 0.82, 1.0))[:, None]
    for (rx0, ry0, rx1, ry1, rz, rc) in rugs:             # area rugs painted onto the floor
        s = (m == 5) & (np.abs(z - rz) < 0.05) & (x > rx0) & (x < rx1) & (y > ry0) & (y < ry1)
        if s.any():
            bd = np.minimum.reduce([x[s] - rx0, rx1 - x[s], y[s] - ry0, ry1 - y[s]]) < 0.35
            base[s] = np.array(rc, float) * (0.95 + 0.06 * hsh(np.floor(x[s] * 8), np.floor(y[s] * 8)))[:, None] * np.where(bd, 0.8, 1.0)[:, None]
    s = np.isin(m, (2, 12, 6))                        # stone: coursed blocks
    if s.any():
        a = np.where(np.abs(n[s, 0]) > 0.5, y[s], x[s]); b = np.where(m[s] == 12, y[s], z[s]) if True else z[s]
        a = np.where(m[s] == 12, x[s], a)
        crs = np.floor(b / 0.45); blk = np.floor((a + hsh(crs) * 3) / (1.2 + 1.4 * hsh(crs, crs)))
        v = 0.84 + 0.26 * hsh(crs, blk); edge = (np.abs((b / 0.45) % 1 - 0.5) > 0.44)
        base[s] *= (v * np.where(edge, 0.8, 1.0))[:, None]
    s = m == 1
    if s.any(): base[s] *= (0.85 + 0.25 * hsh(np.floor(x[s] * 3), np.floor(y[s] * 3)))[:, None]
    s = np.isin(m, (7, 9, 19))                        # wood grain
    if s.any(): base[s] *= (0.9 + 0.12 * np.sin((z[s] + x[s] * 0.3) * 40) * hsh(np.floor(x[s] * 2), np.floor(y[s] * 2)))[:, None]
    # ambient occlusion (screen space)
    Hs, Ws = H * ss, W * ss; dep = np.where(hit, -1.0 / zb, 300.0)
    ao = np.zeros((Hs, Ws))
    for sg in (3 * ss, 10 * ss, 30 * ss): ao += np.clip((dep - ndi.gaussian_filter(dep, sg)) / (0.25 + 0.02 * dep), 0, 1)
    ao = 1 - 0.45 * np.clip(ao / 2.2, 0, 1)
    sky = 0.42 + 0.18 * n[:, 2] + 0.06 * (n @ np.array([-1, 0, 0.0]))          # bias light from the lake side (west)
    inside_px = top[np.clip(((y - 3) / 0.1).astype(int), 0, top.shape[0] - 1), np.clip(((x - 5.5) / 0.1).astype(int), 0, top.shape[1] - 1)] > z + 2
    amb = np.where(inside_px, 0.62 + 0.10 * n[:, 2], 0.55 + 0.25 * n[:, 2])
    sunc = np.array([1.0, 0.93, 0.80]); col = base * ((amb * ao[yy, xx])[:, None] * np.array([0.98, 1.0, 1.03]) + 0.75 * ndl[:, None] * sunc)
    col *= exposure
    img = np.zeros((Hs, Ws, 3))
    # background: sky / lake / far lawn along the view ray
    gy, gx = np.mgrid[0:Hs, 0:Ws]; d = np.stack([(gx + 0.5 - Ws / 2) / fpx, -(gy + 0.5 - Hs / 2) / fpx, np.ones_like(gx, float)], -1) @ np.stack([r, u, f])
    d /= np.linalg.norm(d, axis=-1, keepdims=True); hz = d[..., 2]
    skyc = np.array([150, 190, 225]) * (1 - np.clip(hz, 0, 1)[..., None]) + np.array([95, 140, 205]) * np.clip(hz, 0, 1)[..., None]
    water = np.array([70, 110, 112]) + 25 * np.clip(-hz * 3, 0, 1)[..., None] * np.array([-1, 0, 0])
    lawn = np.array([96, 126, 72])
    bgc = np.where((hz < 0)[..., None], np.where((d[..., 0] < 0)[..., None], water, lawn), skyc)
    img[:] = bgc * 1.15
    img[yy, xx] = col
    # bloom-ish lift on bright windows, gentle tone map
    img = img / 255.0; img = img / (1 + 0.25 * img) * 1.18
    img = np.clip(img ** (1 / 1.05), 0, 1) * 255
    im = Image.fromarray(img.astype(np.uint8)).resize((W, H), Image.LANCZOS)
    if title:
        dr = ImageDraw.Draw(im); fo = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 28)
        dr.rectangle([0, H - 54, W, H], fill=(30, 28, 26)); dr.text((24, H - 44), title, font=fo, fill=(236, 230, 220))
    im.save(out); return im.size
