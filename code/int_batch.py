import sys,time; sys.path.insert(0,'code'); import render_int as R, numpy as np, json
V=np.load('work/int_V.npy'); L=np.load('work/int_L.npy'); top=np.load('work/int_top.npy')
jobs=json.load(open(sys.argv[1])); W=int(sys.argv[2]); ss=int(sys.argv[3])
for j in jobs:
    t=time.time(); R.render(V,L,top,j['eye'],j['look'],j['out'],fov=j.get('fov',70),W=W,H=int(W*0.625),ss=ss,title=j.get('title') if ss>1 else None,rugs=j.get('rugs',[]),props=j.get('props',[]),exposure=j.get('exposure',1.06),sun=tuple(j.get('sun',(235,32))))
    print(j['out'],round(time.time()-t),flush=True)
