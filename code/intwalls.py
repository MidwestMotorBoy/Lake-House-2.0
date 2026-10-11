import pickle, numpy as np, cv2, sys
V=pickle.load(open(f'{sys.argv[1]}/work/vec.pkl','rb'))
X0,X1,Y0,Y1=5.5,149.5,3.0,97.0; F=0.05
W=int(round((X1-X0)/F)); H=int(round((Y1-Y0)/F))
def wallmask(k,dx,L=30):
    img=np.zeros((H,W),np.uint8)
    def P(x,y): return (int(round((x/18-dx-X0)/F)), int(round((y/18-Y0)/F)))
    for s in V[k]['segs']:
        if s[4]>=1.39: cv2.line(img,P(s[0],s[1]),P(s[2],s[3]),1,1)
    for c in V[k]['curves']:
        if c[1]>=1.39 and c[3]:
            p=c[0]
            for i in range(len(p)-1): cv2.line(img,P(*p[i]),P(*p[i+1]),1,1)
    ch=cv2.morphologyEx(img,cv2.MORPH_CLOSE,np.ones((1,L),np.uint8))
    cv_=cv2.morphologyEx(img,cv2.MORPH_CLOSE,np.ones((L,1),np.uint8))
    w=((ch|cv_)>0).astype(np.uint8)
    w=cv2.morphologyEx(w,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    return w.reshape(H//2,2,W//2,2).sum(axis=(1,3))>=2
w11=wallmask('A1.1',4.89); w12=wallmask('A1.2',0.0)
np.save(f'{sys.argv[1]}/work/w11.npy',w11); np.save(f'{sys.argv[1]}/work/w12.npy',w12)
for nm,w in (('w11',w11),('w12',w12)): cv2.imwrite(f'{sys.argv[1]}/img/{nm}.png',255-w.astype(np.uint8)*255)
print(w11.mean(),w12.mean())
