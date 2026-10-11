import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render5 import render
OUT, R = sys.argv[1], sys.argv[2]
asm = sorted(glob.glob(OUT + '/r_asm/*.stl')); opn = sorted(glob.glob(OUT + '/r_open/*.stl')); exp = sorted(glob.glob(OUT + '/r_expl/*.stl'))
jobs = [('01 Lake side', asm, 235, 30, 'Lake Beulah Residence  -  1:75  -  from the lake'),
        ('02 Entry side', asm, 55, 30, 'Entry side'),
        ('03 South-east', asm, 135, 32, 'From the south-east'),
        ('04 North-west', asm, 315, 32, 'From the north-west'),
        ('05 Roofs off', opn, 235, 48, 'Roofs lifted off'),
        ('06 Roofs off entry side', opn, 60, 50, 'Roofs lifted off, entry side'),
        ('07 Exploded', exp, 235, 28, 'Base, floors and roofs'),
        ('08 Plan view', opn, 180, 89.5, 'Plan view, roofs off')]
for name, files, az, el, title in jobs:
    print(name, render(files, f'{R}/{name}.png', az, el, 2000, title), flush=True)
stems = sorted({os.path.basename(p)[2:-4] for p in asm})
for st in stems:
    files = [p for p in asm if os.path.basename(p)[2:-4] == st]
    roof = st[0] == 'C'
    print(st, render(files, f'{R}/pieces/{st}.png', 235, 42 if roof else 38, 1400, st), flush=True)
