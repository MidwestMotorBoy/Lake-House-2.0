"""Lake Beulah Residence - printable massing model built on a voxel grid.
Plan frame: x east, y south (feet, sheet A1.2/A1.3 coordinates). z = project elevation (ft).
"""
import numpy as np, math, sys, pickle
from matplotlib.path import Path
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import zoom, gaussian_filter

G = 0.1                       # grid cell, ft
X0, X1, Y0, Y1 = 5.5, 149.5, 3.0, 97.0
NX = int(round((X1 - X0) / G)); NY = int(round((Y1 - Y0) / G))
ZBASE = 93.5
xi = lambda x: int(round((x - X0) / G))
yi = lambda y: int(round((y - Y0) / G))
xc = X0 + (np.arange(NX) + 0.5) * G
yc = Y0 + (np.arange(NY) + 0.5) * G
XX, YY = np.meshgrid(xc, yc)

def rect(x0, y0, x1, y1):
    m = np.zeros((NY, NX), bool)
    m[max(yi(y0), 0):max(yi(y1), 0), max(xi(x0), 0):max(xi(x1), 0)] = True
    return m

def poly(pts):
    pts = np.asarray(pts, float)
    m = np.zeros((NY, NX), bool)
    i0, i1 = max(xi(pts[:, 0].min()) - 1, 0), min(xi(pts[:, 0].max()) + 1, NX)
    j0, j1 = max(yi(pts[:, 1].min()) - 1, 0), min(yi(pts[:, 1].max()) + 1, NY)
    sub = np.c_[XX[j0:j1, i0:i1].ravel(), YY[j0:j1, i0:i1].ravel()]
    m[j0:j1, i0:i1] = Path(pts).contains_points(sub).reshape(j1 - j0, i1 - i0)
    return m

def arc(cx, cy, r, a0, a1, n=64):
    t = np.linspace(math.radians(a0), math.radians(a1), n)
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in t]

# ------------------------------------------------------------------ levels
RT = 1.35                      # roof slab thickness
P_SPA, P_MAIN = 105.75, 117.09
P_F1, P_N, P_BUNK, P_TOWER = 112.7, 123.14, 125.09, 129.09
P_GOLF, P_SHOW = 125.09, 127.09
SK = 1.35                      # eave skirt thickness

walls = []   # (mask, z0, z1)  -> windows are cut into these
extra = []   # roofs, eaves, terraces, columns

# ------------------------------------------------------------------ walls
F1 = [(40.5, 18.3), (85.5, 18.3), (85.5, 34.4), (90.4, 34.4), (90.4, 37.7), (85.3, 37.7), (85.3, 45.0),
      (90.4, 45.0), (90.4, 48.6), (85.4, 48.6), (85.4, 56.2), (100.7, 56.2), (100.7, 55.7), (145.2, 55.7),
      (145.2, 84.2), (100.7, 84.2), (100.7, 77.1), (47.1, 77.1), (47.1, 88.3), (18.5, 88.3), (18.5, 75.2),
      (18.9, 75.2), (18.9, 61.2), (41.3, 61.2), (41.3, 50.0), (40.5, 50.0)]
mF1 = poly(F1)
spa_only = (XX < 47.1) & (YY > 61.2)
walls.append((mF1, ZBASE, P_SPA))
walls.append((mF1 & ~spa_only, P_SPA, P_F1))
walls.append((rect(80.0, 34.4, 90.4, 48.6), P_F1, 114.85))            # tall foyer

BAY = arc(32.4, 78.25, 13.3, 180 + 46.0, 180 - 45.2, 40)              # north end -> south end (bulges west)
A_W = [(29.4, 61.3), (63.5, 61.3), (63.5, 77.05), (47.1, 77.05), (47.1, 87.9), (23.2, 87.9)] + BAY[::-1] + [(23.1, 68.7), (29.4, 68.7)]
walls.append((poly(A_W), P_SPA, P_MAIN))

N_W = [(41.05, 61.5), (41.05, 45.0)] + arc(69.1, 45.0, 27.9, 180, 180 + 49.1, 48) + \
      [(62.5, 24.0), (62.5, 18.85), (79.55, 18.85), (79.55, 49.9), (74.2, 49.9), (74.2, 61.5)]
walls.append((poly(N_W), P_F1, P_N))
walls.append((rect(62.5, 18.85, 79.55, 37.3), P_N, P_BUNK))
walls.append((rect(57.7, 49.4, 74.2, 65.2), P_F1, P_TOWER))
M_W = [(63.5, 60.7), (84.2, 60.7), (84.2, 59.9), (100.75, 59.9), (100.75, 77.05), (63.5, 77.05)]
walls.append((poly(M_W), P_F1, P_N))
walls.append((rect(100.75, 56.2, 133.2, 84.15), P_F1, P_GOLF))
walls.append((rect(133.2, 57.9, 144.7, 83.0), P_F1, P_SHOW))

# ------------------------------------------------------------------ roofs / eaves
A_R = [(15.38, 65.19), (25.88, 65.19), (25.88, 57.73), (59.43, 57.73), (59.43, 80.57), (50.55, 80.57),
       (50.55, 91.4), (15.38, 91.4)]
mA = poly(A_R)
extra.append((mA, P_MAIN, P_MAIN + RT))
extra.append((mA, P_SPA, P_SPA + SK))                                  # eave band between spa level and main suite
N_R = [(37.51, 64.73), (37.51, 45.6)] + arc(69.1, 45.0, 31.55, 181, 180 + 51.4, 60) + \
      [(49.47, 20.36), (58.93, 20.36), (58.93, 33.32), (83.05, 33.32), (83.05, 53.36), (77.76, 53.36), (77.76, 64.73)]
extra.append((poly(N_R), P_N, P_N + RT))
extra.append((rect(58.93, 15.3, 83.05, 40.8), P_BUNK, P_BUNK + RT))
extra.append((rect(54.35, 45.81, 77.76, 68.77), P_TOWER, P_TOWER + RT))
extra.append((rect(59.43, 56.42, 97.22, 80.57), P_N, P_N + RT))        # billiards / high rollers
extra.append((rect(97.22, 52.65, 136.68, 87.65), P_GOLF, P_GOLF + RT))
extra.append((rect(129.13, 54.4, 148.22, 85.9), P_SHOW, P_SHOW + RT))
LOW = [(37.01, 14.81), (89.09, 14.81), (89.09, 27.86), (83.05, 27.86), (83.05, 52.98), (37.01, 52.98)]
extra.append((poly(LOW), P_F1, P_F1 + SK))                             # great-room roof / balcony deck
extra.append((rect(83.05, 31.31, 100.84, 51.15), 114.85, 116.2))       # entry porch roof
extra.append((rect(83.05, 52.65, 100.7, 56.4), P_F1, P_F1 + SK))       # rear hall roof
extra.append((rect(97.22, 52.15, 148.22, 63.2), P_F1, P_F1 + SK))      # garage eave
for (cx, cy) in ((96.7, 35.3), (96.7, 47.1)):                          # entry columns
    extra.append((rect(cx - 0.7, cy - 0.7, cx + 0.7, cy + 0.7), 99.0, 114.85))
extra.append((rect(33.0, 24.3, 40.6, 45.3), 109.1, 109.8))             # sun louvre
for cy in (24.8, 44.9):
    extra.append((rect(35.1, cy - 0.7, 36.5, cy + 0.7), 97.5, 109.8))
# terraces and lake deck
extra.append((rect(15.8, 7.3, 41.0, 24.3), ZBASE, 100.0))
extra.append((rect(15.8, 24.3, 36.5, 47.3), ZBASE, 98.5))
extra.append((rect(36.5, 24.3, 40.6, 47.3), ZBASE, 100.0))
extra.append((rect(36.5, 47.3, 41.3, 50.0), ZBASE, 98.0))
DECK = [(7.6, 47.3), (41.3, 47.3), (41.3, 61.2), (18.9, 61.2), (18.9, 88.3), (12.3, 88.3), (12.3, 61.2), (7.6, 61.2)]
extra.append((poly(DECK), ZBASE, 96.1))

# ------------------------------------------------------------------ terrain
ctl = []
def P(x, y, z): ctl.append((x, y, z))
for x in (16, 30, 41, 60, 85, 100, 120, 150): P(x, 2, 100); P(x, 12, 100)
for (x, y) in ((86, 25), (95, 25), (120, 25), (150, 25), (92, 40), (120, 40), (150, 40), (102, 52), (125, 52), (140, 52), (150, 50), (88, 52)): P(x, y, 100)
for y in (2, 15, 30, 45, 60, 75, 90, 98): P(4, y, 95); P(9, y, 95)
for (x, y) in ((13, 10), (14, 20), (14, 35), (14, 46), (12, 60), (11, 75), (11, 85), (25, 55), (38, 55)): P(x, y, 95)
for (x, z) in ((14, 97.5), (18, 101.5), (30, 105), (47, 108.5), (68, 112.5), (92, 113), (100, 117.5), (120, 117), (133, 116.3), (150, 116)): P(x, 98, z)
for (x, z) in ((13, 96.5), (18.3, 100.8), (30, 103.5), (47.1, 107.2)): P(x, 89.5, z)
for (x, z) in ((48, 109.6), (58, 110.8), (68, 112.1), (92.5, 112.3), (100, 112.5)): P(x, 78.5, z)
for (x, z) in ((101, 117.5), (117, 117.0), (133, 116.3), (146, 115.6), (150, 115.6)): P(x, 86, z)
for (y, z) in ((57, 103.5), (63, 106.2), (70, 109.4), (77, 112.5), (84, 115.6)): P(146.5, y, z); P(150, y, z)
P(150, 54, 101.5); P(146.5, 53.5, 101)
ctl = np.array(ctl)
gx = np.arange(X0, X1 + 1e-6, 2.0); gy = np.arange(Y0, Y1 + 1e-6, 2.0)
GX, GY = np.meshgrid(gx, gy)
rbf = RBFInterpolator(ctl[:, :2], ctl[:, 2], kernel='linear', smoothing=0.5)
Zc = rbf(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
Zg = zoom(Zc, (NY / Zc.shape[0], NX / Zc.shape[1]), order=1)
Zg = gaussian_filter(Zg, 15)
Zg = np.clip(Zg, 95.0, 118.0)
flat = ((YY < 51.5) & (XX > 85.4)) | ((YY < 18.3) & (XX > 15.8)) | ((YY < 55.7) & (XX > 97) & (XX < 145.2))
Zg[flat] = 100.0
Zg[(XX < 19) & (YY > 47.3) & (YY < 88.3)] = 95.0
Zg[(XX < 47.1) & (YY > 47.3) & (YY < 61.2)] = 95.0
Zg[(XX < 15.8) & (YY < 47.3)] = np.minimum(Zg[(XX < 15.8) & (YY < 47.3)], 95.0 + np.clip((7.3 - YY[(XX < 15.8) & (YY < 47.3)]) / 5.0, 0, 1) * np.clip((XX[(XX < 15.8) & (YY < 47.3)] - 9.0) / 6.8, 0, 1) * 5.0)
ca = rect(69.6, 77.1, 90.5, 84.4); Zg[ca] = np.minimum(Zg[ca], 107.5)   # garden court tiers
cb = rect(71.6, 77.1, 88.6, 81.0); Zg[cb] = 103.5
TSTEP = 0.5
Zq = np.round(Zg / TSTEP) * TSTEP

# ------------------------------------------------------------------ windows: (face, wallcoord, a0, a1, z0, z1, panes)
D_WIN = 4
wins = [
 # west-facing
 ('W', 40.5, 24.5, 44.6, 100.2, 108.9, 5), ('W', 40.5, 24.5, 44.6, 110.2, 112.4, 6),
 ('W', 46.5, 24.6, 35.0, 115.0, 123.0, 4, 6.5), ('W', 41.2, 40.2, 45.0, 115.0, 123.0, 2), ('W', 41.05, 46.2, 57.1, 115.0, 123.0, 3),
 ('W', 29.4, 62.0, 64.8, 108.2, 116.2, 1), ('W', 21.0, 70.0, 86.6, 108.2, 116.2, 7, 4.0),
 ('W', 18.9, 62.8, 72.3, 96.2, 104.0, 3), ('W', 18.5, 76.2, 85.5, 96.2, 104.0, 3), ('W', 41.3, 50.7, 59.9, 96.2, 103.9, 3),
 ('W', 57.7, 50.2, 64.4, 125.4, 128.8, 3), ('W', 62.5, 19.4, 22.4, 114.2, 122.0, 1),
 # north-facing
 ('N', 56.2, 103.6, 130.2, 114.6, 124.9, 6),
 ('N', 55.7, 129.8, 139.9, 100.1, 108.6, 1), ('N', 55.7, 117.7, 127.9, 100.1, 108.6, 1), ('N', 55.7, 105.7, 115.9, 100.1, 108.6, 1),
 ('N', 18.85, 64.8, 77.0, 121.1, 123.0, 3), ('N', 24.0, 55.2, 58.4, 114.2, 122.9, 1),
 ('N', 49.4, 58.8, 73.1, 125.6, 128.8, 3), ('N', 61.3, 32.4, 38.0, 113.4, 115.4, 2), ('N', 61.2, 20.7, 34.2, 96.2, 103.2, 3), ('N', 61.2, 36.9, 39.6, 96.2, 103.0, 1),
 ('N', 59.9, 89.2, 98.1, 115.4, 121.4, 3), ('N', 60.7, 82.0, 83.2, 116.4, 121.2, 1), ('N', 60.7, 79.0, 80.2, 116.4, 121.2, 1),
 ('N', 60.7, 76.0, 77.1, 116.4, 121.2, 1), ('N', 56.2, 90.4, 97.4, 102.4, 108.0, 2), ('N', 68.7, 24.2, 26.6, 108.2, 116.2, 1),
 # east-facing
 ('E', 144.7, 77.5, 78.8, 117.0, 123.9, 1), ('E', 144.7, 74.5, 75.8, 117.0, 123.9, 1), ('E', 144.7, 71.5, 72.8, 117.0, 123.9, 1),
 ('E', 144.7, 68.5, 69.8, 117.0, 123.9, 1), ('E', 144.7, 65.5, 66.8, 117.0, 123.9, 1), ('E', 144.7, 62.5, 63.8, 117.0, 123.9, 1),
 ('E', 85.5, 24.9, 30.4, 103.4, 110.0, 2), ('E', 79.55, 25.4, 30.9, 116.1, 123.0, 2), ('E', 79.55, 41.9, 44.9, 117.1, 121.2, 1),
 ('E', 74.2, 52.7, 56.1, 115.3, 121.1, 1), ('E', 85.4, 50.9, 53.8, 102.1, 108.1, 1), ('E', 89.65, 38.0, 44.7, 109.5, 112.0, 3),
 ('E', 74.2, 50.2, 64.4, 125.4, 128.8, 3),
 # south-facing
 ('S', 87.9, 23.9, 26.8, 108.3, 116.5, 1), ('S', 87.9, 37.1, 40.1, 108.3, 116.5, 1), ('S', 77.05, 55.8, 61.4, 111.7, 116.5, 2),
 ('S', 77.05, 67.8, 69.4, 115.4, 121.4, 1), ('S', 77.05, 70.9, 72.4, 115.4, 121.4, 1), ('S', 77.05, 73.9, 75.4, 115.4, 121.4, 1),
 ('S', 77.05, 76.9, 78.4, 115.4, 121.4, 1), ('S', 77.05, 79.9, 81.4, 115.4, 121.4, 1),
 ('S', 77.1, 72.5, 83.1, 103.7, 110.3, 3), ('S', 77.1, 85.0, 87.6, 103.7, 108.3, 1),
 ('S', 83.0, 134.0, 144.0, 116.1, 124.1, 1), ('S', 65.2, 58.8, 73.1, 125.6, 128.8, 3),
]


# =====================================================================  1:75 take-apart ("dollhouse") build
from scipy import ndimage as ndi
OUT = sys.argv[1]
w11 = np.load(OUT + '/w11.npy'); w12 = np.load(OUT + '/w12.npy')
KEY = 0.5
PD = 0.55            # magnet pocket depth (2.2 mm at 1:75)
ero = lambda m, n: ndi.binary_erosion(m, structure=np.ones((3, 3), bool), iterations=n) if n > 0 else m
dil = lambda m, n: ndi.binary_dilation(m, structure=np.ones((3, 3), bool), iterations=n) if n > 0 else m

# ---- z grid: quarter-foot levels plus every special level
zset = set(np.round(np.arange(ZBASE, 131.01, 0.25), 2).tolist())
special = [P_SPA, P_MAIN, P_F1, P_N, P_BUNK, P_TOWER, P_GOLF, P_SHOW, 114.85, 116.2, 109.1, 109.8, 96.1, 98.5]
special += [112.0]
for p in list(special): special += [p + RT, p + KEY, p + SK, p + PD, p - PD, p + 0.3, p - 0.3]
for w in wins: special += [w[4], w[5]]
zset.update(np.round(special, 2).tolist())
zs = np.array(sorted(zset)); NZ = len(zs) - 1
def K(z):
    k = int(np.argmin(abs(zs - z))); assert abs(zs[k] - z) < 1e-6, ('z level missing', z); return k
zmid = (zs[:-1] + zs[1:]) / 2
print('grid', NX, NY, NZ)

# ---- zones (plan footprints of the upper-floor trays)
TOWER = rect(57.7, 49.4, 74.2, 65.2)
BUNK = rect(62.5, 18.85, 79.55, 37.3)
FN = poly(N_W) | TOWER
FM = poly(M_W) & ~TOWER
GOLF = rect(100.75, 56.2, 133.2, 84.15); SHOW = rect(133.2, 57.9, 144.7, 83.0)
FG = GOLF | SHOW
FA = poly(A_W) & ~(FN | FM | FG)
FOY = rect(80.0, 34.4, 90.4, 48.6)
UP = FN | FM | FG
plateA = (mA | FA) & ~UP
plateN = (poly(LOW) | FN | rect(83.05, 52.65, 100.7, 56.4)) & ~dil(FOY, 1) & ~(FM | FG)
plateG = FG | (rect(100.85, 52.15, 148.22, 63.2) & ~FM)
plateG = plateG | (rect(100.7, 55.7, 145.2, 84.2) & ~FM)
plateN = plateN | (rect(84.2, 56.2, 100.7, 61.0) & mF1 & ~FM & ~FG)
plateN = plateN | ((rect(83.0, 27.8, 85.6, 34.5) | rect(83.0, 48.5, 85.6, 52.7) | rect(74.1, 52.9, 84.3, 60.8)) & mF1 & ~FM & ~FG & ~dil(FOY, 1))
plateA = plateA | (rect(18.8, 61.1, 26.0, 65.3) & mF1)
gap = mF1 & ~plateA & ~plateN & ~plateG & ~FM & ~dil(FOY, 1)
lab_g, n_g = ndi_label(gap) if False else (None, 0)
import scipy.ndimage as _n
lg, ng = _n.label(gap)
print('uncovered ground-floor cells:', int(gap.sum()), 'in', ng, 'patches', flush=True)
for q in range(1, ng + 1):
    yy, xx = np.nonzero(lg == q)
    if len(yy) > 150: print('   patch', len(yy), 'x', round(xc[xx.min()], 1), round(xc[xx.max()], 1), 'y', round(yc[yy.min()], 1), round(yc[yy.max()], 1), flush=True)
WELL = rect(62.7, 50.0, 73.9, 64.9) & ero(TOWER, 7)                         # main stairwell, open through the upper floor
plateN = plateN & ~WELL
T_ST, T_SD, SKIN = 8, 7, 3          # stone wall, siding wall, coloured outer skin (cells)

class Piece:
    def __init__(s, name, zlo, zhi):
        s.name = name; s.kb = K(zlo); s.ke = K(zhi); s.lab = np.zeros((s.ke - s.kb, NY, NX), np.uint8)
    def sl(s, z0, z1): return s.lab[max(K(z0) - s.kb, 0):max(K(z1) - s.kb, 0)]
    def at(s, k): return s.lab[k - s.kb]
    def has(s, k): return s.kb <= k < s.ke
    def fill(s, m, z0, z1, L): s.sl(z0, z1)[:, m] = L
    def clear(s, m, z0, z1): s.sl(z0, z1)[:, m] = 0
    def walls(s, F, t, z0, z1, plan=None):
        ring = F & ~ero(F, t); skin = F & ~ero(F, SKIN)
        s.fill(ring, z0, z1, 2); s.fill(skin, z0, z1, 3)
        if plan is not None: s.fill(plan & ero(F, t - 1) & ~ring, z0, z1, 2)
    def key(s, z, region):
        k = K(z) - s.kb; top = (s.lab[k - 1] > 0) & (s.lab[k] == 0) & region
        s.lab[k:K(z + KEY) - s.kb][:, top] = 2

def cutwin(P, face, wc, a0, a1, z0, z1, n, reach=2.2, T=7, door=False, mull=0.5):
    k0, k1 = max(K(z0) - P.kb, 0), min(K(z1) - P.kb, P.ke - P.kb)
    if k1 <= k0: return 0
    if face in 'WE':
        j0, j1 = yi(a0), yi(a1); c0, c1 = xi(wc - reach), xi(wc + reach)
        view = lambda A: A[k0:k1, j0:j1, c0:c1]; lat = yc[j0:j1]
    else:
        i0, i1 = xi(a0), xi(a1); c0, c1 = yi(wc - reach), yi(wc + reach)
        view = lambda A: A[k0:k1, c0:c1, i0:i1].transpose(0, 2, 1); lat = xc[i0:i1]
    fl = (lambda A: A[:, :, ::-1]) if face in 'ES' else (lambda A: A)
    sub = fl(view(P.lab)); sol = sub > 0
    if not sol.any(): return 0
    first = sol.argmax(axis=2); has = sol.any(axis=2)
    rel = np.arange(sol.shape[2])[None, None, :] - first[:, :, None]
    ok = has[:, :, None] & sol & (rel >= 0)
    keep = np.zeros(len(lat), bool)
    for t in range(1, n): keep |= abs(lat - (a0 + (a1 - a0) * t / n)) < mull / 2
    ok[:, keep, :] = False
    outer = ok & (rel < 2); glass = ok & (rel >= 2) & (rel < 4); inner = ok & (rel >= 4) & (rel < T)
    sub[outer] = 0
    sub[glass] = 3 if door else 4
    if not door: sub[inner] = 0
    return int(outer.sum())

pieces = []
# ---------------------------------------------------------------- upper-floor trays
A = Piece('Main suite floor', P_SPA, P_MAIN + KEY)
A.fill(plateA, P_SPA, 107.0, 3); A.fill(plateA & ero(FA, SKIN), P_SPA, 107.0, 2)
A.walls(FA, T_SD, 107.0, P_MAIN, plan=w12)
N = Piece('North bedrooms floor', P_F1, P_TOWER + KEY)
N.fill(plateN, P_F1, 114.0, 3); N.fill(plateN & ero(FN, SKIN), P_F1, 114.0, 2)
N.walls(FN, T_SD, 114.0, P_N, plan=w12)
N.walls(BUNK, T_SD, P_N, P_BUNK); N.walls(TOWER, T_SD, P_N, P_TOWER)
N.clear(WELL, P_F1, P_N)
M = Piece('Billiards and high rollers floor', 112.0, P_N + KEY)
M.fill(FM, 112.0, 113.0, 3); M.fill(ero(FM, SKIN), 112.0, 113.0, 2)
M.walls(FM, T_SD, 113.0, P_N, plan=w12)
Gp = Piece('Golf bar and showroom floor', 112.0, P_SHOW + KEY)
Gp.fill(plateG, 112.0, 113.0, 3); Gp.fill(ero(FG, SKIN), 112.0, 113.0, 2)
Gp.walls(FG, T_SD, 113.0, P_GOLF, plan=w12)
Gp.walls(SHOW, T_SD, P_GOLF, P_SHOW)
Gp.fill(ero(SHOW, T_SD), 113.0, 116.0, 2)                                   # raised showroom floor
Gp.fill(rect(122.8, 65.7, 125.0, 75.4), 113.0, 116.5, 3)                    # bar
Gp.fill(rect(129.3, 62.0, 131.3, 78.5) & ero(FG, T_SD), 113.0, 116.0, 3)    # back bar
door = rect(99.6, 60.4, 101.9, 75.0)                                        # billiards is open to the golf room
M.clear(door, 113.0, 122.0); Gp.clear(door, 113.0, 122.0)
# stair #2: landing in the stair tower (107) up 9 risers east to the billiards floor (113), with the door through the tower wall
S2Y0, S2Y1 = 61.3, 64.6
N.clear(rect(73.3, S2Y0, 74.5, S2Y1), P_F1, 121.0)
M.clear(rect(74.2, S2Y0, 75.1, S2Y1), 112.0, 121.0)
M.clear(rect(74.2, S2Y0, 80.6, S2Y1), 112.0, 113.0)
# main suite hall opens onto the stair balcony (second-floor plan: opening at the west end of the balcony)
A.clear(rect(56.9, 62.4, 58.0, 64.7), 107.0, 114.0); N.clear(rect(57.5, 62.4, 58.7, 64.7), P_F1, 114.0)
# Mech #3: floor is at the balcony level (107) with a 16 ft ceiling, entered by door 25 from the balcony - not at the bedroom level
M3I = rect(53.8, 49.8, 57.9, 60.7); M3O = rect(53.2, 49.2, 58.4, 61.2); M3D = rect(57.7, 57.2, 58.8, 59.9)
N.clear(M3I, P_F1, P_N); N.clear(M3D, P_F1, 114.0)
N.clear(rect(58.4, 55.8, 62.9, 64.45), P_F1, 114.0)                          # balcony in front of the elevator is open above
# ---------------------------------------------------------------- roofs / lids
def lid(name, m, z0, z1):
    p = Piece(name, z0, z1); p.fill(m, z0, z1, 3); return p
porch = lid('Entry porch roof', rect(79.7, 31.31, 100.84, 51.15), 114.85, 116.2)
rA = lid('Main suite roof', (mA | rect(59.43, 57.73, 63.5, 80.57)) & ~dil(UP, 1), P_MAIN, P_MAIN + RT)
rN = lid('North bedrooms roof', poly(N_R) & ~dil(BUNK, 1) & ~dil(TOWER, 1), P_N, P_N + RT)
rB = lid('Bunk room roof', rect(58.93, 15.3, 83.05, 40.8), P_BUNK, P_BUNK + RT)
rT = lid('Stair tower roof', rect(54.35, 45.81, 77.76, 68.77), P_TOWER, P_TOWER + RT)
rM = lid('Billiards roof', rect(59.43, 56.42, 97.22, 80.57) & ~dil(TOWER, 1) & ~dil(poly(N_R), 1), P_N, P_N + RT)
rG = lid('Golf and bar roof', rect(97.22, 52.65, 136.68, 87.65) & ~dil(SHOW, 1), P_GOLF, P_GOLF + RT)
rS = lid('Showroom roof', rect(129.13, 54.4, 148.22, 85.9), P_SHOW, P_SHOW + RT)
uppers = [A, N, M, Gp, porch, rA, rN, rM, rB, rT, rG, rS]
# lowest upper-piece surface over each column (for capping the hill so pieces can drop in)
zmin_up = np.full((NY, NX), 999.0)
for p in uppers:
    anyk = p.lab > 0; first = anyk.argmax(axis=0); has = anyk.any(axis=0)
    zmin_up = np.where(has, np.minimum(zmin_up, zs[first + p.kb]), zmin_up)
zmin_up = ndi.minimum_filter(zmin_up, size=7)

# ---------------------------------------------------------------- base
B = Piece('Base', ZBASE, 118.5)
FALL = mF1
floor96 = FALL & (YY > 50.0) & (XX < 67.5)
stone2d = ~((XX < 58.8) & (YY > 50.0)) & ~((YY < 19.6) & (XX > 48.1) & (XX < 81.8))
terr = np.minimum(Zq, np.floor((zmin_up - 0.01) / 0.25) * 0.25)
solid_foot = np.zeros((NY, NX), bool)
terr_items = [(rect(15.8, 7.3, 41.0, 24.3), 100.0, 2), (rect(15.8, 24.3, 36.5, 47.3), 98.5, 2), (rect(36.5, 24.3, 40.6, 47.3), 100.0, 2),
              (rect(36.5, 47.3, 41.3, 50.0), 98.0, 2), (poly(DECK), 96.0, 3)]
for k in range(B.kb, B.ke):
    if zmid[k] < terr.max(): B.at(k)[(terr > zmid[k]) & ~FALL] = 1
B.lab[0] = 1
for m, zt, L in terr_items:
    mm = m & ~FALL
    for k in range(K(zt)):
        B.at(k)[mm & (B.at(k) == 0)] = L
    B.fill(mm, zt - 0.5, zt, L)                                             # paved surface colour even where the yard is level with it
B.fill(FALL & ~floor96, ZBASE, 100.0, 2); B.fill(floor96, ZBASE, 96.0, 2)
B.lab[0][FALL] = 1                                                          # ground colour on the underside
ztop = np.full((NY, NX), P_F1)
ztop[plateA] = P_SPA; ztop[FM | FG] = 112.0
ring = FALL & ~ero(FALL, T_ST); skin = FALL & ~ero(FALL, SKIN); inter = w11 & ero(FALL, T_ST - 1) & ~ring
for zt in (P_SPA, 112.0, P_F1):
    sel = ztop == zt
    B.fill(ring & sel, 96.0, zt, 2); B.fill(skin & sel & stone2d, 95.0, zt, 2); B.fill(skin & sel & ~stone2d, 95.0, zt, 3)
    B.fill(inter & sel & floor96, 96.0, zt, 2); B.fill(inter & sel & ~floor96, 100.0, zt, 2)
fr = FOY & FALL
B.fill(fr & ring, P_F1, 114.85, 2); B.fill(fr & skin, P_F1, 114.85, 2); B.fill(fr & inter, P_F1, 114.85, 2)
B.fill(FOY & ~ero(FOY, T_ST) & FALL, 100.0, 114.85, 2)                      # foyer flank walls
B.clear(rect(85.9, 38.0, 89.6, 44.7), 100.0, 114.85)                        # open porch recess in front of the door
# foyer per the first-floor plan: open to the house on the west, no wall on the house wall line, front door at the east face
B.clear(rect(79.9, 37.7, 81.3, 44.9), 100.0, 114.85); B.clear(rect(84.3, 37.7, 85.9, 44.9), 100.0, 114.85)
B.fill(rect(89.4, 37.7, 89.9, 44.9), 109.0, 114.85, 2)                      # header over the front door
B.fill(rect(86.7, 43.0, 89.7, 43.3), 100.0, 109.0, 3)                       # pivot door, standing open
for (cx, cy) in ((96.7, 35.3), (96.7, 47.1)): B.fill(rect(cx - 0.7, cy - 0.7, cx + 0.7, cy + 0.7), 100.0, 114.85, 3)
B.fill(rect(33.0, 24.3, 40.6, 45.3), 109.1, 109.8, 3)
for cy in (24.8, 44.9): B.fill(rect(35.1, cy - 0.7, 36.5, cy + 0.7), 98.5, 109.8, 3)
# fixed features
B.fill(rect(65.3, 26.3, 77.4, 30.3), 100.0, 103.0, 3)                       # kitchen island
B.fill(rect(49.0, 19.1, 78.0, 21.5), 100.0, 103.0, 3)                       # kitchen wall counter
B.clear(rect(22.1, 62.2, 29.3, 74.5), 94.25, 96.0)                          # spa pool
B.fill(rect(22.1, 62.2, 29.3, 74.5), 94.25, 94.5, 4)                        # water
def _put(P, m, z0, z1, L):
    v = P.sl(z0, z1); sub = v[:, m]; v[:, m] = np.where(sub == 0, L, sub)
q25 = lambda z: round(z * 4) / 4
RISE = 7.0 / 12
# plan walls inside the stair hall are only low walls under the flights: take them out so nothing divides the stair
B.clear(rect(63.9, 50.0, 67.5, 59.4), 96.0, P_F1 + KEY); B.clear(rect(67.5, 50.0, 73.5, 64.9), 100.0, P_F1 + KEY)
B.clear(rect(63.9, 59.4, 67.5, 64.9), 100.0, P_F1 + KEY)
for i in range(12):                                                         # main stair, per the plans
    # east flight: first floor (100) at the north end, rising south to the landing at 107
    _put(B, rect(68.0, 50.4 + i * 0.75, 73.5, 50.4 + (i + 1) * 0.75), 100.0, q25(100.0 + (i + 1) * RISE), 3)
    # west flight: landing (107) at the south end, rising north to the upper hall at 114
    zt = q25(107.0 + (i + 1) * RISE)
    _put(B, rect(63.9, 59.4 - (i + 1) * 0.75, 68.0, 59.4 - i * 0.75), zt - 1.25, zt, 3)
_put(B, rect(57.9, 59.4, 74.0, 65.0), 106.5, 107.0, 3)                      # landing / balcony at the main-suite level
_put(B, rect(68.7, 59.4, 69.3, 64.9), 100.0, 106.5, 2)                      # storage wall under the landing (carries it)
for i in range(7):                                                          # flight down to the spa level under the west flight
    _put(B, rect(63.9, 50.4 + i * 0.64, 67.5, 50.4 + (i + 1) * 0.64), 96.0, q25(100.0 - (i + 1) * (4.0 / 7)), 3)
B.clear(rect(73.5, S2Y0, 74.5, S2Y1), 107.0, P_F1 + KEY)
B.clear(rect(57.7, 61.5, 58.7, 64.9), 107.0, P_F1 + KEY)
B.clear(M3I, 107.0, P_F1 + KEY)
_put(B, M3O, 106.5, 107.0, 2); _put(B, M3O & ~M3I, 107.0, P_F1, 2)
_put(B, rect(58.4, 55.8, 63.9, 59.4), 106.5, 107.0, 3)                      # balcony runs on in front of the elevator to door 25
B.clear(M3D, 107.0, P_F1 + KEY)
for k, zt in enumerate((107.75, 108.25, 109.0, 109.75, 110.25, 111.0, 111.75, 112.25)):
    r = rect(74.2 + k * 0.8, S2Y0, 74.2 + (k + 1) * 0.8, S2Y1)
    B.clear(r, zt, P_F1 + KEY)
    v = B.sl(100.0, zt); sub = v[:, r]; v[:, r] = np.where(sub == 0, 3, sub); B.sl(zt - 0.25, zt)[:, r] = 3
B.fill(rect(73.5, S2Y0, 74.2, S2Y1), 106.5, 107.0, 3)
# ---- terrace stairs, per F1.1 / A1.1
def _step(r, zt, L=2):                       # solid step: clear above the tread, fill below it
    r = r & ~FALL; B.clear(r, zt, 104.0); v = B.sl(ZBASE + 0.25, zt); sub = v[:, r]; v[:, r] = np.where(sub == 0, L, sub); B.sl(zt - 0.25, zt)[:, r] = L
# lake stair: 7 risers from the first-floor level (100) at the east end, heading WEST down to the lake deck (96)
_step(rect(35.7, 46.4, 41.3, 49.6), 100.0)
for k, zt in enumerate((99.5, 98.75, 98.25, 97.75, 97.25, 96.5)):
    _step(rect(34.7 - k * 1.0, 46.4, 35.7 - k * 1.0, 49.6), zt)
_step(rect(24.4, 46.4, 29.7, 47.3), 96.0, 3)                                 # deck runs up to the foot of the stair
# fire terrace (98.5) down 4 risers heading south to the lake deck, at the south-west corner
for k, zt in enumerate((98.0, 97.25, 96.75)):
    _step(rect(20.0, 43.2 + k * 1.05, 24.4, 43.2 + (k + 1) * 1.05), zt)
_step(rect(20.0, 46.35, 24.4, 47.3), 96.0, 3)
# grill terrace (100) down 3 risers heading south to the fire terrace, at the north-west corner
_step(rect(20.1, 24.3, 24.2, 25.25), 99.5); _step(rect(20.1, 25.25, 24.2, 26.2), 99.0)
for i, zt in enumerate((113.5, 114.25, 114.75, 115.5, 116.0)):              # bar up to the showroom
    _put(Gp, rect(127.0 + i * 1.1, 57.0, 128.1 + i * 1.1, 60.9), 113.0, zt, 3)
_put(Gp, rect(132.5, 57.0, 134.0, 60.9), 113.0, 116.0, 2)
Gp.clear(rect(132.6, 58.4, 134.4, 60.6), 116.0, 123.0)
# inlaid lettering on the lake deck walkway, reading correctly when viewed from the lake (west) looking at the house
from PIL import Image as _I, ImageDraw as _D, ImageFont as _F
tx0, ty0, tx1, ty1 = 12.7, 63.8, 18.5, 88.0
tw, th = yi(ty1) - yi(ty0), xi(tx1) - xi(tx0)          # line length runs north-south, lines stack east-west
timg = _I.new('L', (tw, th), 0); td = _D.Draw(timg)
lines = ['Lance\u2019s Barbie', 'Dream House']
fpath = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
fs = 60
while True:
    fnt = _F.truetype(fpath, fs)
    boxes = [td.textbbox((0, 0), l, font=fnt) for l in lines]
    if max(b[2] - b[0] for b in boxes) <= tw and len(lines) * int(fs * 1.25) <= th: break
    fs -= 1
lh = int(fs * 1.25); y0 = (th - lh * len(lines)) // 2
for n, (l, b) in enumerate(zip(lines, boxes)):
    td.text(((tw - (b[2] - b[0])) // 2 - b[0], y0 + n * lh - b[1] + (lh - (b[3] - b[1])) // 2), l, font=fnt, fill=255)
U = np.array(timg) > 128
tmask = np.zeros((NY, NX), bool); tmask[yi(ty0):yi(ty0) + tw, xi(tx0):xi(tx0) + th] = U[::-1, :].T
B.fill(tmask, 95.75, 96.0, 2)
print('lettering font px', fs, 'cells', int(tmask.sum()), flush=True)
# ---- furniture (layout from the architect's furniture plans F1.1 / F1.2; simple contemporary shapes)
def fbox(P, x0, y0, x1, y1, zf, h, L=3, z0=0.0): _put(P, rect(x0, y0, x1, y1), zf + z0, zf + h, L)
def _back(x0, y0, x1, y1, side, t):
    return {'N': (x0, y0, x1, y0 + t), 'S': (x0, y1 - t, x1, y1), 'W': (x0, y0, x0 + t, y1), 'E': (x1 - t, y0, x1, y1)}[side]
def sofa(P, x0, y0, x1, y1, zf, backs, t=0.7, hb=2.5):
    fbox(P, x0, y0, x1, y1, zf, 0.75, 3)
    cx0, cy0, cx1, cy1 = x0 + 0.2, y0 + 0.2, x1 - 0.2, y1 - 0.2
    for b in backs:
        bx = _back(x0, y0, x1, y1, b, t); fbox(P, *bx, zf, hb, 3)
        if b == 'N': cy0 = y0 + t
        if b == 'S': cy1 = y1 - t
        if b == 'W': cx0 = x0 + t
        if b == 'E': cx1 = x1 - t
    fbox(P, cx0, cy0, cx1, cy1, zf, 1.5, 2, z0=0.75)
def chair(P, cx, cy, zf, back, s=2.4): sofa(P, cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2, zf, [back], t=0.5)
def dchair(P, cx, cy, zf, back, s=1.6):
    fbox(P, cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2, zf, 1.5, 3)
    fbox(P, *_back(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2, back, 0.4), zf, 3.0, 3)
def bed(P, x0, y0, x1, y1, zf, head):
    fbox(P, x0, y0, x1, y1, zf, 1.0, 3)
    fbox(P, x0 + 0.2, y0 + 0.2, x1 - 0.2, y1 - 0.2, zf, 1.75, 2, z0=1.0)
    fbox(P, *_back(x0, y0, x1, y1, head, 0.4), zf, 3.5, 3)
def oval(P, cx, cy, rx, ry, zf, h, L=3): _put(P, ((XX - cx) / rx) ** 2 + ((YY - cy) / ry) ** 2 <= 1.0, zf, zf + h, L)
a1 = lambda x: x - 4.30                       # F1.1 sheet -> model frame
bx2 = lambda x: x + 2.60; by2 = lambda y: y - 9.25   # F1.2 sheet -> model frame
def B1(fn, x0, y0, x1, y1, *r, **k): fn(B, a1(x0), y0, a1(x1), y1, *r, **k)
# ground floor and terraces (base)
B1(fbox, 32.0, 16.0, 40.5, 19.8, 100.0, 2.5)                                  # grill terrace table
for cx in (34.0, 38.4):
    dchair(B, a1(cx), 15.0, 100.0, 'N'); dchair(B, a1(cx), 20.8, 100.0, 'S')
dchair(B, a1(31.0), 17.9, 100.0, 'W'); dchair(B, a1(41.5), 17.9, 100.0, 'E')
B1(fbox, 34.0, 9.6, 38.0, 11.6, 100.0, 3.0)                                   # grill
# fire terrace (F1.1): gas fire pit table, U-shaped sectional around it, three loungers on the lake side
B1(fbox, 28.9, 31.5, 32.2, 37.8, 98.5, 1.5, 2); B1(fbox, 30.0, 32.7, 31.0, 36.7, 98.5, 1.75, 3, z0=1.5)
B1(sofa, 29.0, 26.0, 35.0, 29.0, 98.5, ['N']); B1(sofa, 35.0, 26.0, 37.9, 43.2, 98.5, ['E']); B1(sofa, 29.0, 40.3, 35.0, 43.2, 98.5, ['S'])
for ly in (29.7, 33.2, 36.7): B1(fbox, 24.8, ly, 26.3, ly + 2.9, 98.5, 1.0, 3)
B1(sofa, 51.8, 24.3, 54.8, 27.5, 100.0, ['W']); B1(sofa, 58.8, 24.3, 61.8, 27.5, 100.0, ['E'])
B1(fbox, 56.0, 25.2, 58.0, 27.2, 100.0, 1.25)
B1(sofa, 55.0, 32.2, 66.6, 35.8, 100.0, ['N']); B1(sofa, 63.8, 35.8, 66.6, 43.5, 100.0, ['E'])
B1(fbox, 57.8, 30.9, 63.8, 32.0, 100.0, 2.5)                                  # sofa console
B1(fbox, 57.0, 37.5, 62.0, 40.5, 100.0, 1.25)                                 # coffee table
B1(fbox, 52.6, 33.5, 54.6, 35.5, 100.0, 1.25)
chair(B, a1(50.5), 37.3, 100.0, 'W', 2.8); chair(B, a1(50.5), 42.5, 100.0, 'W', 2.8)
B1(fbox, 67.2, 33.5, 68.8, 40.5, 100.0, 2.5)                                  # media console
B1(fbox, 47.6, 46.9, 61.3, 48.3, 100.0, 3.0)                                  # south built-in
B1(fbox, 73.5, 34.3, 77.4, 44.2, 100.0, 2.5)                                  # dining table
for cy in (35.9, 38.4, 40.9, 43.3):
    dchair(B, a1(72.4), cy, 100.0, 'W'); dchair(B, a1(78.5), cy, 100.0, 'E')
for cx in (72.0, 74.2, 76.5, 78.7): fbox(B, a1(cx) - 0.6, 30.2, a1(cx) + 0.6, 31.4, 100.0, 2.5)   # island stools
B1(fbox, 95.2, 56.9, 99.2, 58.0, 100.0, 1.5)                                  # rear hall bench
B1(bed, 78.5, 68.6, 84.8, 75.2, 100.0, 'S')
B1(fbox, 76.3, 73.6, 78.3, 75.3, 100.0, 2.0); B1(fbox, 85.0, 73.6, 87.0, 75.3, 100.0, 2.0)
B1(fbox, 78.6, 65.3, 84.6, 66.6, 100.0, 2.5)
chair(B, a1(91.0), 72.8, 100.0, 'E', 2.8)
B1(fbox, 33.0, 66.0, 39.6, 67.9, 96.0, 1.25); B1(fbox, 33.0, 69.0, 39.6, 70.9, 96.0, 1.25)   # spa loungers
B1(fbox, 41.3, 71.6, 50.3, 73.8, 96.0, 1.5); B1(fbox, 47.6, 68.3, 50.3, 71.6, 96.0, 3.0)     # sauna benches
B1(fbox, 63.5, 68.0, 70.3, 69.8, 96.0, 1.5); B1(fbox, 63.5, 73.3, 70.3, 75.0, 96.0, 1.5)     # steam benches
_put(B, (abs(YY - 85.55) <= 1.3 * np.clip(1 - ((XX - a1(34.2)) / 7.3) ** 2, 0, 1)) & (abs(XX - a1(34.2)) < 7.3), 96.0, 97.5, 3)   # canoe
B1(fbox, 26.0, 53.2, 34.6, 56.8, 96.0, 2.5)                                   # lake deck table
for cx in (28.0, 32.4):
    dchair(B, a1(cx), 52.2, 96.0, 'N'); dchair(B, a1(cx), 57.8, 96.0, 'S')
dchair(B, a1(25.0), 55.0, 96.0, 'W'); dchair(B, a1(35.6), 55.0, 96.0, 'E')
for cy in (51.0, 55.3, 60.7, 64.6): chair(B, a1(20.3), cy, 96.0, 'W', 2.6)    # lake chairs
# main suite floor
def A2(fn, x0, y0, x1, y1, *r, **k): fn(A, bx2(x0), by2(y0), bx2(x1), by2(y1), *r, **k)
A2(bed, 24.4, 84.0, 30.2, 90.6, 107.0, 'E')
A2(fbox, 29.8, 81.3, 31.4, 83.3, 107.0, 2.0); A2(fbox, 29.8, 91.3, 31.4, 93.3, 107.0, 2.0)
A2(fbox, 22.7, 85.2, 24.1, 90.0, 107.0, 1.5)
chair(A, bx2(18.6), by2(84.3), 107.0, 'W'); chair(A, bx2(18.6), by2(90.4), 107.0, 'W')
A2(fbox, 18.6, 86.9, 20.4, 88.7, 107.0, 1.25)
A2(fbox, 24.5, 95.0, 30.5, 96.3, 107.0, 2.5)
A2(fbox, 31.6, 74.9, 33.4, 85.2, 107.0, 5.0)                                  # dressing wardrobes
A2(fbox, 38.4, 74.9, 50.1, 76.6, 107.0, 5.0); A2(fbox, 38.4, 83.7, 50.1, 85.4, 107.0, 5.0)
A2(fbox, 58.0, 74.9, 60.3, 85.2, 107.0, 2.5); chair(A, bx2(56.2), by2(80.0), 107.0, 'W', 2.2)   # office desk wall + chair
# north bedrooms floor
def N2(fn, x0, y0, x1, y1, *r, **k): fn(N, bx2(x0), by2(y0), bx2(x1), by2(y1), *r, **k)
N2(bed, 52.9, 36.4, 59.3, 42.6, 114.0, 'E')
N2(fbox, 57.7, 33.3, 59.4, 35.2, 114.0, 2.0); N2(fbox, 57.7, 44.3, 59.4, 46.2, 114.0, 2.0)
N2(fbox, 51.0, 37.2, 52.4, 41.8, 114.0, 1.5)
chair(N, bx2(47.5), by2(36.2), 114.0, 'N'); chair(N, bx2(44.6), by2(40.6), 114.0, 'W')
N2(fbox, 43.4, 44.2, 49.3, 45.7, 114.0, 2.5)
N2(bed, 43.9, 61.2, 50.2, 67.2, 114.0, 'E')
N2(fbox, 48.6, 59.0, 50.3, 60.8, 114.0, 2.0); N2(fbox, 48.6, 67.5, 50.3, 69.3, 114.0, 2.0)
N2(fbox, 42.1, 61.8, 43.5, 66.7, 114.0, 1.5)
for (y0, y1) in ((28.7, 33.4), (40.6, 45.2)):                                # queen bunks, two tiers
    N2(bed, 69.4, y0, 75.9, y1, 114.0, 'E')
    N2(fbox, 69.4, y0, 75.9, y1, 114.0, 4.75, 3, z0=4.0); N2(fbox, 69.6, y0 + 0.2, 75.7, y1 - 0.2, 114.0, 5.5, 2, z0=4.75)
    N2(fbox, 69.4, y0, 69.9, y0 + 0.5, 114.0, 4.0, 3, z0=1.0); N2(fbox, 69.4, y1 - 0.5, 69.9, y1, 114.0, 4.0, 3, z0=1.0)
chair(N, bx2(64.2), by2(36.2), 114.0, 'N'); chair(N, bx2(64.2), by2(41.6), 114.0, 'S')
oval(N, bx2(64.8), by2(39.0), 0.9, 0.9, 114.0, 1.5)
N2(fbox, 60.0, 32.2, 61.4, 45.4, 114.0, 2.5)
# billiards / high rollers floor
def M2(fn, x0, y0, x1, y1, *r, **k): fn(M, bx2(x0), by2(y0), bx2(x1), by2(y1), *r, **k)
oval(M, bx2(71.6), by2(80.4), 3.4, 2.2, 113.0, 2.5)
for (cx, cy, b) in ((69.0, 77.5, 'N'), (71.6, 77.2, 'N'), (74.2, 77.5, 'N'), (69.0, 83.4, 'S'), (71.6, 83.7, 'S'), (74.2, 83.4, 'S'), (67.2, 80.4, 'W'), (76.0, 80.4, 'E')):
    dchair(M, bx2(cx), by2(cy), 113.0, b)
M2(fbox, 63.3, 78.3, 64.8, 81.2, 113.0, 2.5)
chair(M, bx2(64.7), by2(76.4), 113.0, 'W'); chair(M, bx2(64.7), by2(84.6), 113.0, 'W')
M2(fbox, 79.9, 79.6, 81.3, 84.3, 113.0, 2.5)
M2(fbox, 87.0, 74.8, 94.9, 78.6, 113.0, 2.75); M2(fbox, 87.3, 75.1, 94.6, 78.3, 113.0, 2.75, 2, z0=2.5)   # billiards table with light bed
for x0 in (87.1, 89.1, 91.0, 93.0): M2(fbox, x0, 70.3, x0 + 1.7, 71.8, 113.0, 1.25)
M2(fbox, 85.3, 83.8, 96.9, 85.4, 113.0, 5.0)                                  # wine display wall
# golf / bar floor
def G2(fn, x0, y0, x1, y1, *r, **k): fn(Gp, bx2(x0), by2(y0), bx2(x1), by2(y1), *r, **k)
G2(sofa, 100.0, 67.2, 115.1, 70.8, 113.0, ['N'])
for cy in (76.0, 78.5, 81.0, 83.4): fbox(Gp, bx2(119.3), by2(cy) - 0.7, bx2(120.7), by2(cy) + 0.7, 113.0, 2.5)
G2(fbox, 117.1, 88.3, 119.3, 92.3, 113.0, 3.5)                                # pinball
# locating keys on wall tops
inner = ero(FALL, 4)
B.key(P_SPA, plateA & inner); B.key(P_F1, plateN & inner); B.key(112.0, (FM | FG) & inner); B.key(114.85, FOY & inner)
A.key(P_MAIN, ero(FA, 4)); N.key(P_N, ero(FN, 4) & ~BUNK & ~TOWER); N.key(P_BUNK, ero(BUNK, 4)); N.key(P_TOWER, ero(TOWER, 4))
M.key(P_N, ero(FM, 4)); Gp.key(P_GOLF, ero(FG, 4) & ~SHOW); Gp.key(P_SHOW, ero(SHOW, 4))

# ---------------------------------------------------------------- windows (every piece; only bites where that piece has a wall)
order = [B] + uppers
for w in wins:
    isdoor = (w[0] == 'N' and w[1] == 55.7) or (w[0] == 'S' and w[1] == 83.0)
    tot = 0
    for p in (B, A, N, M, Gp):
        tot += cutwin(p, *w[:7], reach=(w[7] if len(w) > 7 else 2.2), T=(T_ST if p is B else T_SD), door=isdoor)
    if tot == 0: print('WARNING window cut nothing', w)

# ---------------------------------------------------------------- magnet posts and pockets (6 x 2 mm discs)
MAGS = []
def magnets(lower, upper, z, region, zf, n, tag):
    kz = K(z)
    occ = (lower.lab[K(zf + 0.5) - lower.kb:kz - lower.kb] > 0).any(axis=0)
    glass = (lower.lab == 4).any(axis=0)
    upsolid = ero(upper.at(kz) > 0, 10)
    free = ero(region, T_ST + 1) & ~dil(occ, 1) & ~dil(glass, 12) & upsolid & ~dil(rect(22.1, 62.2, 29.3, 74.5), 3)
    fit = ndi.binary_erosion(free, structure=np.ones((23, 23), bool))
    if not fit.any(): print('  no room for a magnet post:', tag, flush=True); return
    score = ndi.uniform_filter(occ.astype(float), 33); score = score / max(score[fit].max(), 1e-9)
    jj, ii = np.nonzero(fit); sc = score[jj, ii]; chosen = []
    for t in range(n):
        if not chosen: q = int(np.argmax(sc))
        else:
            dmin = np.min([np.hypot(jj - cj, ii - ci) for cj, ci in chosen], axis=0)
            if dmin.max() < 60: break
            q = int(np.argmax(dmin * (0.35 + sc)))
        chosen.append((jj[q], ii[q]))
    for cj, ci in chosen:
        box = np.zeros((NY, NX), bool); box[cj - 13:cj + 14, ci - 13:ci + 14] = True
        box &= ero(region | dil(region, 2), SKIN)
        col = lower.lab[:kz - lower.kb]
        col[:, box] = np.where(col[:, box] == 0, 2, col[:, box])
        disk = ((XX - (X0 + (ci + 0.5) * G)) ** 2 + (YY - (Y0 + (cj + 0.5) * G)) ** 2) <= 0.8 ** 2
        lower.lab[K(z - PD) - lower.kb:kz - lower.kb][:, disk] = 0
        upper.lab[kz - upper.kb:K(z + PD) - upper.kb][:, disk] = 0
        MAGS.append((tag, round(X0 + ci * G, 1), round(Y0 + cj * G, 1)))
    print(f'  {tag}: {len(chosen)} magnet pairs', flush=True)
# magnets(B, A, P_SPA, mF1 & plateA, 96.0, 4, 'base / main suite floor')
# magnets(B, N, P_F1, mF1 & plateN & ~dil(FOY, 2), 100.0, 4, 'base / north bedrooms floor')
# magnets(B, M, 112.0, mF1 & FM, 100.0, 3, 'base / billiards floor')
# magnets(B, Gp, 112.0, mF1 & FG, 100.0, 4, 'base / golf floor')
# magnets(B, porch, 114.85, FOY & mF1, 100.0, 2, 'base / porch roof')
# magnets(A, rA, P_MAIN, FA, 107.0, 4, 'main suite / roof')
# magnets(N, rN, P_N, FN & ~dil(BUNK, 1) & ~dil(TOWER, 1), 114.0, 4, 'north bedrooms / roof')
# magnets(N, rB, P_BUNK, BUNK, 114.0, 2, 'bunk room / roof')
# magnets(N, rT, P_TOWER, TOWER, 114.0, 2, 'stair tower / roof')
# magnets(M, rM, P_N, FM, 113.0, 3, 'billiards / roof')
# magnets(Gp, rG, P_GOLF, GOLF, 113.0, 3, 'golf and bar / roof')
# magnets(Gp, rS, P_SHOW, SHOW, 116.0, 2, 'showroom / roof')
# ---- small in-wall magnets (2 mm dia x 1 mm discs): pockets in a slight thickening at wall corners, no posts
MD = 0.3                       # pocket depth, 1.2 mm
def wallmag(lower, upper, z, region, n, tag):
    kz = K(z); kl = kz - lower.kb
    below = lower.lab[kl - 1] > 0
    glass = (lower.lab == 4).any(axis=0) | (upper.lab == 4).any(axis=0)
    band = ero(region, 5) & ~ero(region, 6) & below & ~dil(glass, 4) & ero(upper.at(kz) > 0, 4)
    wallish = (lower.lab[max(kl - 8, 0):kl] > 0).all(axis=0)
    band &= wallish
    if not band.any(): print('  no wall spot for a magnet:', tag, flush=True); return
    score = ndi.uniform_filter(wallish.astype(float), 17); jj, ii = np.nonzero(band); sc = score[jj, ii] / max(score[jj, ii].max(), 1e-9)
    chosen = []
    for t in range(n):
        if not chosen: q = int(np.argmax(sc))
        else:
            dmin = np.min([np.hypot(jj - cj, ii - ci) for cj, ci in chosen], axis=0)
            if dmin.max() < 50: break
            q = int(np.argmax(dmin * (0.35 + sc)))
        chosen.append((jj[q], ii[q]))
    for cj, ci in chosen:
        boss = np.zeros((NY, NX), bool); boss[cj - 5:cj + 5, ci - 5:ci + 5] = True; boss &= region
        col = lower.lab[:kl]; col[:, boss] = np.where(col[:, boss] == 0, 2, col[:, boss])
        flat = np.zeros((NY, NX), bool); flat[cj - 7:cj + 7, ci - 7:ci + 7] = True; flat &= region
        lower.lab[kl:K(z + KEY) - lower.kb][:, flat] = 0                   # no key here, so the faces meet flat
        pk = np.zeros((NY, NX), bool); pk[cj - 3:cj + 3, ci - 3:ci + 3] = True
        lower.lab[K(z - MD) - lower.kb:kl][:, pk] = 0
        upper.lab[kz - upper.kb:K(z + MD) - upper.kb][:, pk] = 0
        MAGS.append((tag, round(X0 + ci * G, 1), round(Y0 + cj * G, 1)))
    print(f'  {tag}: {len(chosen)} magnet pairs', flush=True)
wallmag(B, A, P_SPA, mF1 & plateA, 4, 'base / main suite floor')
wallmag(B, N, P_F1, mF1 & plateN & ~dil(FOY, 2), 4, 'base / north bedrooms floor')
wallmag(B, M, 112.0, mF1 & FM, 3, 'base / billiards floor')
wallmag(B, Gp, 112.0, mF1 & FG, 4, 'base / golf floor')
wallmag(B, porch, 114.85, FOY & mF1, 2, 'base / porch roof')
wallmag(A, rA, P_MAIN, FA, 4, 'main suite / roof')
wallmag(N, rN, P_N, FN & ~dil(BUNK, 1) & ~dil(TOWER, 1), 4, 'north bedrooms / roof')
wallmag(N, rB, P_BUNK, BUNK, 3, 'bunk room / roof')
wallmag(N, rT, P_TOWER, TOWER, 3, 'stair tower / roof')
wallmag(M, rM, P_N, FM, 3, 'billiards / roof')
wallmag(Gp, rG, P_GOLF, GOLF, 4, 'golf and bar / roof')
wallmag(Gp, rS, P_SHOW, SHOW, 3, 'showroom / roof')
print('magnet pairs total:', len(MAGS), flush=True)
# ---------------------------------------------------------------- make later pieces clear earlier ones (grooves + clearances)
for a in range(1, len(order)):
    p = order[a]
    for q in order[:a]:
        for k in range(max(p.kb, q.kb), min(p.ke, q.ke)):
            qo = q.at(k) > 0
            if not qo.any(): continue
            pl = p.at(k); hit = (pl > 0) & dil(qo, 1)
            if hit.any(): pl[hit] = 0
for p in order:                                                             # drop loose crumbs
    cc, n = ndi.label(p.lab > 0)
    if n > 1:
        sizes = ndi.sum(p.lab > 0, cc, index=np.arange(1, n + 1)); big = sizes.argmax() + 1
        small = (cc != big) & (cc > 0)
        print(f'{p.name}: {n} bodies, removing {int(small.sum())} cells ({small.sum() / (p.lab > 0).sum():.3%}); second largest {np.sort(sizes)[-2]:.0f}', flush=True)
        p.lab[small] = 0
    del cc
# ---------------------------------------------------------------- checks
dbl = 0
for k in range(NZ):
    c = np.zeros((NY, NX), np.uint8)
    for p in order:
        if p.has(k): c += (p.at(k) > 0)
    dbl += int((c > 1).sum())
print('cells occupied by two pieces (must be 0):', dbl, flush=True)
top_prev = np.full((NY, NX), -1); blocked = 0
for p in order:
    anyk = p.lab > 0; has = anyk.any(axis=0)
    lowest = np.where(has, anyk.argmax(axis=0) + p.kb, 10 ** 6)
    bad = has & (top_prev > lowest)
    if bad.any(): print(f'  {p.name}: {int(bad.sum())} columns sit under an earlier piece', flush=True); blocked += int(bad.sum())
    if bad.any():
        yy_, xx_ = np.nonzero(bad); print('     at x', round(xc[xx_.min()], 1), round(xc[xx_.max()], 1), 'y', round(yc[yy_.min()], 1), round(yc[yy_.max()], 1), 'lowest z', zs[lowest[bad].min()], 'earlier top z', zs[top_prev[bad].max() + 1], flush=True)
    hi = np.where(has, p.ke - 1 - anyk[::-1].argmax(axis=0), -1)
    top_prev = np.maximum(top_prev, hi)
print('columns where a piece cannot drop straight in:', blocked, flush=True)
import pickle
meta = []
S75 = 304.8 / 75
for p in order:
    nz = np.nonzero(p.lab.any(axis=(1, 2)))[0]; ny = np.nonzero(p.lab.any(axis=(0, 2)))[0]; nx = np.nonzero(p.lab.any(axis=(0, 1)))[0]
    k0, k1, j0, j1, i0, i1 = nz[0], nz[-1] + 1, ny[0], ny[-1] + 1, nx[0], nx[-1] + 1
    np.save(f'{OUT}/d_{len(meta):02d}.npy', p.lab[k0:k1, j0:j1, i0:i1])
    meta.append(dict(name=p.name, k0=int(k0 + p.kb), k1=int(k1 + p.kb), j0=int(j0), j1=int(j1), i0=int(i0), i1=int(i1)))
    print(f'{p.name:34s} {(i1 - i0) * G * S75:6.0f} x {(j1 - j0) * G * S75:5.0f} x {(zs[k1 + p.kb] - zs[k0 + p.kb]) * S75:5.0f} mm   colours {sorted(np.unique(p.lab).tolist())}', flush=True)
pickle.dump(dict(meta=meta, zs=zs, NY=NY, NX=NX), open(f'{OUT}/d_meta.pkl', 'wb'))
