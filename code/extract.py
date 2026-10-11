import pdfplumber, sys, pickle, glob
D="/mnt/user-data/uploads/Final Drawings"
out={}
for k in sys.argv[2:]:
    f=glob.glob(f"{D}/*{k} *.pdf")[0]
    pg=pdfplumber.open(f).pages[0]
    segs=[]  # (x0,y0,x1,y1,linewidth,dashed)
    for l in pg.lines:
        segs.append((l['x0'],l['top'],l['x1'],l['bottom'],l.get('linewidth',0),bool(l.get('dash')) and l.get('dash')!=([],0), l['pts'] if 'pts' in l else None))
    curves=[(c['pts'],c.get('linewidth',0),c.get('fill'),c.get('stroke'),c.get('dash')) for c in pg.curves]
    rects=[(r['x0'],r['top'],r['x1'],r['bottom'],r.get('linewidth',0),r.get('fill'),r.get('stroke')) for r in pg.rects]
    words=[(w['x0'],w['top'],w['x1'],w['bottom'],w['text'],w.get('upright',True)) for w in pg.extract_words(keep_blank_chars=False,use_text_flow=False,extra_attrs=['upright'])]
    out[k]=dict(segs=segs,curves=curves,rects=rects,words=words,size=(pg.width,pg.height))
    print(k,len(segs),len(curves),len(rects),len(words))
pickle.dump(out,open(sys.argv[1],'wb'))
