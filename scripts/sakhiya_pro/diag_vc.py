import sys, json, numpy as np
sys.path.insert(0,'/work/sp'); import sp_common as C
import soundfile as sf
F=json.load(open('/work/sp/data/ia_filtered.json'))[:60:2]
a=[];b=[]
for x in F:
    t=x['text']
    ho,_=C.kriti().read_conf([f"/work/sp/data/ia_wav/{x['id']}.wav"])[0]
    hv,_=C.kriti().read_conf([f"/work/sp/data/ia_vc/{x['id']}.wav"])[0]
    a.append(C.cer(t,ho)); b.append(C.cer(t,hv))
print('orig cer mean/median',np.mean(a),np.median(a)); print('vc cer',np.mean(b),np.median(b))
print([round(v,2) for v in a[:15]]); print([round(v,2) for v in b[:15]])
x=F[3]; print(x['text']); print(C.kriti().read_conf([f"/work/sp/data/ia_wav/{x['id']}.wav"])[0][0])
