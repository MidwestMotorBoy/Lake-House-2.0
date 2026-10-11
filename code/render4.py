"""Orthographic z-buffer renderer for a binary STL. usage: render.py in.stl out.png az el [width]"""
import numpy as np, sys, math
from PIL import Image
f=sys.argv[1]; out=sys.argv[2]; az=math.radians(float(sys.argv[3])); el=math.radians(float(sys.argv[4])); Wd=int(sys.argv[5]) if len(sys.argv)>5 else 1500
import glob
dt=np.dtype([('n','<f4',3),('v','<f4',(3,3)),('a','<u2')])
PAL={'1':(127,145,96),'2':(210,200,180),'3':(75,58,45),'4':(141,182,208)}
Vs=[];Cs=[]
for p in sorted(glob.glob(f+'/*.stl')):
    d=np.fromfile(p,dtype=dt,offset=84); Vs.append(d['v'].astype(np.float64)); Cs.append(np.tile(np.array(PAL[p.split('/')[-1][0]],float),(len(d),1)))
V=np.concatenate(Vs); TC=np.concatenate(Cs)
# camera basis: view direction from camera toward scene
cz=np.array([-math.cos(el)*math.sin(az),-math.cos(el)*math.cos(az),-math.sin(el)])   # forward
up=np.array([0,0,1.0]); cx=np.cross(cz,up); cx/=np.linalg.norm(cx); cy=np.cross(cx,cz)
P=np.stack([V@cx,V@cy,V@cz],axis=-1)       # screen x, screen y(up), depth (larger = farther)
mn=P.reshape(-1,3).min(0); mx=P.reshape(-1,3).max(0)
sc=(Wd-40)/(mx[0]-mn[0]); Hh=int((mx[1]-mn[1])*sc)+40
X=(P[...,0]-mn[0])*sc+20; Y=(mx[1]-P[...,1])*sc+20; Z=P[...,2]
nrm=np.cross(V[:,1]-V[:,0],V[:,2]-V[:,0]); nrm/=np.maximum(np.linalg.norm(nrm,axis=1),1e-12)[:,None]
face=nrm@(-cz)
L=np.array([-0.45,-0.35,0.82]); L/=np.linalg.norm(L)
L=L[0]*cx+L[1]*(-cz)*0+np.array([0,0,1])*0  # placeholder (overwritten below)
light=np.array([math.sin(az-0.9)*0.55, math.cos(az-0.9)*0.55, 0.75]); light/=np.linalg.norm(light)
sh=0.35+0.65*np.clip(nrm@light,0,1)
zb=np.full((Hh,Wd),np.inf); col=np.full((Hh,Wd),1.0); cim=np.zeros((Hh,Wd,3))
idx=np.nonzero(face>1e-6)[0]
for t in idx:
    x=X[t]; y=Y[t]; z=Z[t]
    x0=max(int(math.floor(x.min())),0); x1=min(int(math.ceil(x.max()))+1,Wd); y0=max(int(math.floor(y.min())),0); y1=min(int(math.ceil(y.max()))+1,Hh)
    if x1<=x0 or y1<=y0: continue
    den=(y[1]-y[2])*(x[0]-x[2])+(x[2]-x[1])*(y[0]-y[2])
    if abs(den)<1e-12: continue
    gx,gy=np.meshgrid(np.arange(x0,x1)+0.5,np.arange(y0,y1)+0.5)
    a=((y[1]-y[2])*(gx-x[2])+(x[2]-x[1])*(gy-y[2]))/den
    b=((y[2]-y[0])*(gx-x[2])+(x[0]-x[2])*(gy-y[2]))/den
    c=1-a-b
    m=(a>=-1e-6)&(b>=-1e-6)&(c>=-1e-6)
    if not m.any(): continue
    zz=a*z[0]+b*z[1]+c*z[2]
    sub=zb[y0:y1,x0:x1]; upd=m&(zz<sub-1e-7)
    sub[upd]=zz[upd]; col[y0:y1,x0:x1][upd]=sh[t]; cim[y0:y1,x0:x1][upd]=TC[t]
bg=~np.isfinite(zb)
zf=np.where(bg,np.nanmax(zb[~bg]),zb)
e=np.abs(np.diff(zf,axis=0,prepend=zf[:1]))+np.abs(np.diff(zf,axis=1,prepend=zf[:,:1]))
edge=np.clip(e/1.2,0,1)
img=np.clip(col*(1-0.55*edge),0,1)
rgb=cim*(0.25+0.85*img[...,None]); rgb=np.clip(rgb,0,255)
rgb[bg]=255
Image.fromarray(rgb.astype(np.uint8)).save(out); print(Wd,Hh,len(idx))
