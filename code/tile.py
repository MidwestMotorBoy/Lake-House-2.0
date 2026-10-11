# usage: tile.py SHEET x0 y0 x1 y1 (pts) scale(px/pt) out [gridft]
import sys,glob,subprocess
from PIL import Image,ImageDraw,ImageFont
k=sys.argv[1]; x0,y0,x1,y1=map(float,sys.argv[2:6]); sc=float(sys.argv[6]); out=sys.argv[7]
g=float(sys.argv[8]) if len(sys.argv)>8 else 10
D="/mnt/user-data/uploads/Final Drawings"
f=glob.glob(f"{D}/*{k} *.pdf")[0]
dpi=72*sc
subprocess.run(["pdftoppm","-r",str(dpi),"-x",str(int(x0*sc)),"-y",str(int(y0*sc)),"-W",str(int((x1-x0)*sc)),"-H",str(int((y1-y0)*sc)),"-png","-singlefile",f,out[:-4]],check=True)
im=Image.open(out).convert("RGB"); d=ImageDraw.Draw(im)
try: fo=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",13)
except: fo=None
import math
fx=math.ceil(x0/18/g)*g
while fx*18<x1:
    px=(fx*18-x0)*sc; d.line([(px,0),(px,im.height)],fill=(255,120,120),width=1); d.text((px+2,2),f"{fx:g}",fill=(220,0,0),font=fo); d.text((px+2,im.height-16),f"{fx:g}",fill=(220,0,0),font=fo); fx+=g
fy=math.ceil(y0/18/g)*g
while fy*18<y1:
    py=(fy*18-y0)*sc; d.line([(0,py),(im.width,py)],fill=(120,120,255),width=1); d.text((2,py+2),f"{fy:g}",fill=(0,0,220),font=fo); d.text((im.width-30,py+2),f"{fy:g}",fill=(0,0,220),font=fo); fy+=g
im.save(out); print(im.size)
