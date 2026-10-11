"""Voxel volume (non-uniform z) -> conforming watertight triangle mesh."""
import numpy as np, struct, bisect, time
def mesh_bool(vol, zs, SCALE=150.0):
    NZ, NY, NX = vol.shape
    G = 0.1; t0 = time.time()

    def runs2d(B):
        """B: 2D bool. Runs along axis 1. Returns arrays (row, start, end)."""
        pad = np.zeros((B.shape[0], B.shape[1] + 2), np.int8); pad[:, 1:-1] = B
        d = np.diff(pad, axis=1)
        r0, s = np.nonzero(d == 1); r1, e = np.nonzero(d == -1)
        return r0, s, e

    def merge(rows, s, e):
        """Merge identical runs on consecutive rows -> list of (row0,row1,s,e)."""
        res = []; active = {}; prev_row = -10
        n = len(rows); idx = 0
        while idx < n:
            r = rows[idx]; j = idx
            new = {}
            while j < n and rows[j] == r:
                key = (int(s[j]), int(e[j]))
                new[key] = active[key] if (key in active and prev_row == r - 1) else r
                j += 1
            # close runs that did not continue
            for key, rs in active.items():
                if key not in new or new[key] != rs:
                    res.append((rs, prev_row + 1, key[0], key[1]))
            active = new; prev_row = r; idx = j
        for key, rs in active.items():
            res.append((rs, prev_row + 1, key[0], key[1]))
        return res
    quads = []   # (axis, c, a0, a1, b0, b1, sign)   z: c=k, a=j, b=i ; x: c=i, a=j, b=k ; y: c=j, a=i, b=k
    # --- horizontal faces
    empty = np.zeros((NY, NX), bool)
    for k in range(NZ + 1):
        below = vol[k - 1] if k > 0 else empty
        above = vol[k] if k < NZ else empty
        for sign, m in ((1, below & ~above), (-1, above & ~below)):
            if not m.any(): continue
            for (j0, j1, i0, i1) in merge(*runs2d(m)):
                quads.append((2, k, j0, j1, i0, i1, sign))
    nh = len(quads)
    # --- vertical faces, merged through z
    def vertical(axis):
        open_ = {1: {}, -1: {}}
        for k in range(NZ + 1):
            if k < NZ:
                a = vol[k] if axis == 0 else vol[k].T          # axis0: boundaries between columns i ; rows = j
                pad = np.zeros((a.shape[0], a.shape[1] + 2), bool); pad[:, 1:-1] = a
                L = pad[:, :-1]; R = pad[:, 1:]
                masks = {1: (L & ~R).T, -1: (~L & R).T}       # shape (ncol+1, nrow): runs along the row direction
            for sign in (1, -1):
                new = {}
                if k < NZ:
                    c, s, e = runs2d(masks[sign])
                    old = open_[sign]
                    for q in range(len(c)):
                        key = (int(c[q]), int(s[q]), int(e[q]))
                        new[key] = old.pop(key, k)
                for key, ks in open_[sign].items():
                    quads.append((axis, key[0], key[1], key[2], ks, k, sign))
                open_[sign] = new
    vertical(0); vertical(1)
    print('quads', len(quads), 'horizontal', nh, 'time', round(time.time() - t0, 1))

    # --- corners in (i,j,k)
    def corners(q):
        ax, c, a0, a1, b0, b1, sg = q
        if ax == 2:   # c=k, a=j, b=i
            P = [(b0, a0, c), (b1, a0, c), (b1, a1, c), (b0, a1, c)]
        elif ax == 0: # c=i, a=j, b=k   order (j,k)
            P = [(c, a0, b0), (c, a1, b0), (c, a1, b1), (c, a0, b1)]
        else:         # c=j, a=i, b=k   order (k,i)
            P = [(a0, c, b0), (a0, c, b1), (a1, c, b1), (a1, c, b0)]
        return P if sg > 0 else P[::-1]

    allc = np.array([p for q in quads for p in corners(q)], np.int64)
    V = np.unique(allc, axis=0)
    lines = [{}, {}, {}]
    def build(ax):
        o = [a for a in range(3) if a != ax]
        order = np.lexsort((V[:, ax], V[:, o[1]], V[:, o[0]]))
        W = V[order]
        key = W[:, o[0]] * 100000 + W[:, o[1]]
        brk = np.nonzero(np.diff(key))[0] + 1
        st = np.r_[0, brk]; en = np.r_[brk, len(W)]
        d = {}
        vals = W[:, ax]
        for a, b in zip(st, en):
            if b - a > 2: d[int(key[a])] = vals[a:b].tolist()
        lines[ax] = d
    for ax in range(3): build(ax)
    print('verts', len(V), 'time', round(time.time() - t0, 1))

    tris = []      # float coords triples in ijk space
    bedges = []    # boundary polygon edges as codes
    code = lambda p: (p[0] * (NY + 1) + p[1]) * (NZ + 1) + p[2]
    nsplit = 0
    for q in quads:
        C = corners(q)
        poly = []
        for t in range(4):
            a = C[t]; b = C[(t + 1) % 4]
            poly.append(a)
            ax = 0 if a[0] != b[0] else (1 if a[1] != b[1] else 2)
            o = [x for x in range(3) if x != ax]
            lst = lines[ax].get(a[o[0]] * 100000 + a[o[1]])
            if lst is None: continue
            lo, hi = (a[ax], b[ax]) if a[ax] < b[ax] else (b[ax], a[ax])
            i0 = bisect.bisect_right(lst, lo); i1 = bisect.bisect_left(lst, hi)
            if i1 > i0:
                mid = lst[i0:i1]
                if a[ax] > b[ax]: mid = mid[::-1]
                for m in mid:
                    p = list(a); p[ax] = m; poly.append(tuple(p))
        n = len(poly)
        for t in range(n): bedges.append((code(poly[t]), code(poly[(t + 1) % n])))
        if n == 4:
            tris.append((poly[0], poly[1], poly[2])); tris.append((poly[0], poly[2], poly[3]))
        else:
            nsplit += 1
            cx = (C[0][0] + C[2][0]) / 2; cy = (C[0][1] + C[2][1]) / 2; cz = (C[0][2] + C[2][2]) / 2
            cen = (cx, cy, cz)
            for t in range(n): tris.append((cen, poly[t], poly[(t + 1) % n]))
    print('tris', len(tris), 'split quads', nsplit, 'time', round(time.time() - t0, 1))

    # --- manifold check on polygon boundary edges
    E = np.array(bedges, np.int64)
    fw = E[:, 0] * (2 ** 31) + E[:, 1]; bw = E[:, 1] * (2 ** 31) + E[:, 0]
    uf, cf = np.unique(fw, return_counts=True); ub, cb = np.unique(bw, return_counts=True)
    open_edges = np.setdiff1d(uf, ub).size + np.setdiff1d(ub, uf).size
    print('open edges', open_edges, 'edges used >1 per direction (non-manifold contacts)', int((cf > 1).sum()))

    # --- to physical mm, flip y, write STL
    T = np.array(tris, float)                      # (n,3,3) ijk (k is index, may be fractional for centers)
    S = 304.8 / SCALE
    kk = T[:, :, 2]
    zf = np.interp(kk, np.arange(len(zs)), zs)
    Pm = np.empty_like(T)
    Pm[:, :, 0] = T[:, :, 0] * G * S
    Pm[:, :, 1] = (NY - T[:, :, 1]) * G * S
    Pm[:, :, 2] = (zf - zs[0]) * S
    Pm = Pm[:, ::-1, :]                             # y flip reverses orientation -> restore
    n = np.cross(Pm[:, 1] - Pm[:, 0], Pm[:, 2] - Pm[:, 0])
    volm = (Pm[:, 0] * n).sum() / 6.0
    ln = np.linalg.norm(n, axis=1); print('degenerate tris', int((ln < 1e-9).sum()), 'volume cm3', round(volm / 1000, 1))
    n = n / np.maximum(ln, 1e-12)[:, None]

    return Pm, n, dict(open_edges=int(open_edges), nonmanifold=int((cf > 1).sum()), volume_cm3=float(volm/1000))
def write_stl(path, Pm, n, name=b'model'):
    rec = np.zeros(len(Pm), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    rec['n'] = n; rec['v'] = Pm
    with open(path, 'wb') as f:
        f.write(name.ljust(80, b' ')[:80]); f.write(struct.pack('<I', len(Pm))); f.write(rec.tobytes())
