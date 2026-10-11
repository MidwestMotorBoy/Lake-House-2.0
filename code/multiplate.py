import numpy as np, glob, os, zipfile, json, re, sys
S=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COL={1:'Ground',2:'Stone and interior',3:'Siding and roofs',4:'Glass'}
dt=np.dtype([('n','<f4',3),('v','<f4',(3,3)),('a','<u2')])
stems=sorted({os.path.basename(p).rsplit(' - ',1)[0] for p in glob.glob(S+'/out/stl/*.stl')})
only=sys.argv[2:] ; 
if only: stems=[s for s in stems if s.split()[0] in only]
n=len(stems); v=np.sqrt(n); cols=int(round(v))+(1 if v>round(v) else 0)
SX,SY=330*1.2,320*1.2
proj=json.load(open(os.path.dirname(os.path.abspath(__file__))+'/project_settings.config'))
TX,TY=140.0,255.0
proj['wipe_tower_x']=[str(TX)]*n; proj['wipe_tower_y']=[str(TY)]*n
res=[];items=[];cfg=['<?xml version="1.0" encoding="UTF-8"?>\n<config>\n'];plates=[];oid=0
def mesh_xml(i,Pm):
    V=np.round(Pm.reshape(-1,3),4);U,inv=np.unique(V,axis=0,return_inverse=True);T=inv.reshape(-1,3)
    return ''.join([f'<object id="{i}" type="model"><mesh><vertices>',''.join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x,y,z in U),'</vertices><triangles>',''.join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a,b,c in T),'</triangles></mesh></object>'])
for pi,stem in enumerate(stems):
    parts=[]
    for L in (1,2,3,4):
        f=f'{S}/out/stl/{stem} - {L} {COL[L]}.stl'
        if os.path.exists(f): parts.append((L,np.fromfile(f,dtype=dt,offset=84)['v'].astype(np.float64)))
    allv=np.concatenate([p.reshape(-1,3) for _,p in parts]); lo=allv.min(0); hi=allv.max(0)
    col,row=pi%cols,pi//cols
    tx=col*SX+175-(lo[0]+hi[0])/2; ty=-row*SY+8-lo[1]; tz=-lo[2]
    ids=[]
    for L,Pm in parts:
        oid+=1; ids.append((oid,L)); res.append(mesh_xml(oid,Pm))
    oid+=1; top=oid
    res.append(f'<object id="{top}" type="model"><components>'+''.join(f'<component objectid="{i}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>' for i,_ in ids)+'</components></object>')
    items.append(f'<item objectid="{top}" transform="1 0 0 0 1 0 0 0 1 {tx:.4f} {ty:.4f} {tz:.4f}" printable="1"/>')
    cfg.append(f'  <object id="{top}">\n    <metadata key="name" value="{stem}"/>\n    <metadata key="extruder" value="{ids[0][1]}"/>\n')
    for i,L in ids: cfg.append(f'    <part id="{i}" subtype="normal_part">\n      <metadata key="name" value="{L} {COL[L]}"/>\n      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n      <metadata key="extruder" value="{L}"/>\n    </part>\n')
    cfg.append('  </object>\n')
    plates.append(f'  <plate>\n    <metadata key="plater_id" value="{pi+1}"/>\n    <metadata key="plater_name" value="{stem}"/>\n    <metadata key="locked" value="false"/>\n    <model_instance>\n      <metadata key="object_id" value="{top}"/>\n      <metadata key="instance_id" value="0"/>\n      <metadata key="identify_id" value="{1000+pi}"/>\n    </model_instance>\n  </plate>\n')
    print(stem, np.round(hi-lo,0))
model=('<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:BambuStudio="http://schemas.bambulab.com/package/2021">'
 '<metadata name="Application">BambuStudio-02.08.02.61</metadata><metadata name="BambuStudio:3mfVersion">1</metadata><metadata name="Title">Lake Beulah dollhouse</metadata><resources>'+''.join(res)+'</resources><build>'+''.join(items)+'</build></model>')
CT='<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/><Default Extension="config" ContentType="application/xml"/></Types>'
RELS='<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>'
with zipfile.ZipFile(sys.argv[1],'w',zipfile.ZIP_DEFLATED) as z:
    z.writestr('[Content_Types].xml',CT); z.writestr('_rels/.rels',RELS); z.writestr('3D/3dmodel.model',model)
    z.writestr('Metadata/model_settings.config',''.join(cfg)+''.join(plates)+'</config>\n'); z.writestr('Metadata/project_settings.config',json.dumps(proj,indent=4))
print(n,cols,os.path.getsize(sys.argv[1]))
