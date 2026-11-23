#!/usr/bin/env python3
"""Native-pronunciation eval for Sakhiya-Pro v2: 24 held-out sentences packed with Nepal-native hard words
(schwa retention, nasal diphthongs, conjuncts, long/short vowels) + English tech loanwords + Sanskrit.
Metric per sentence: CER (sound-folded, validators.pron_cer), TARGET-WORD RECALL in the ASR transcript, Hindi leaks, UTMOS, spk-sim, clicks.
usage: eval_pron.py <ckpt_dir> <tag> [num_step=32] [guidance=2.0]"""
import sys, os, json, re, numpy as np, torch, soundfile as sf
sys.path.insert(0, '/work/sp'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sp_common as C, validators as V, nep_norm
from omnivoice.models.omnivoice import OmniVoice
S = [
 ('p01', 'बाट हिँड्दा आज समय छैन भन्दै उनी हतार हतार गए।', ['बाट', 'आज', 'समय', 'छैन']),
 ('p02', 'होइन, मसँग त्यो किताब छैन तर तिमीसँग छ।', ['होइन', 'मसँग', 'छैन', 'तिमीसँग']),
 ('p03', 'काठमाडौँ उपत्यकामा आज बिहानै भारी वर्षा भयो।', ['काठमाडौँ', 'आज', 'वर्षा']),
 ('p04', 'गाउँमा पानी आउँछ भन्ने खबर सुनेर सबै खुसी भए।', ['गाउँमा', 'आउँछ', 'खबर']),
 ('p05', 'सरकारले यस वर्ष शिक्षा क्षेत्रमा तीन अर्ब रुपैयाँ विनियोजन गरेको छ।', ['सरकारले', 'शिक्षा', 'क्षेत्रमा', 'अर्ब']),
 ('p06', 'प्रधानमन्त्रीले आज राष्ट्रिय सभालाई सम्बोधन गर्दै विकासको प्राथमिकता दोहोर्‍याए।', ['प्रधानमन्त्रीले', 'राष्ट्रिय', 'सम्बोधन']),
 ('p07', 'मौसम विभागका अनुसार भोलि पहाडी भेगमा हिमपातको सम्भावना छ।', ['मौसम', 'विभाग', 'पहाडी', 'हिमपात']),
 ('p08', 'बुबाले छोरीलाई भन्नुभयो, खाना खाएर मात्र स्कुल जानू है।', ['बुबाले', 'छोरीलाई', 'खाना', 'स्कुल']),
 ('p09', 'उनले आफ्नो सपना पूरा गर्न धेरै संघर्ष गरिन्।', ['सपना', 'पूरा', 'संघर्ष']),
 ('p10', 'हिजोको बैठकमा मूल्य वृद्धि र बेरोजगारीको विषयमा गम्भीर छलफल भयो।', ['बैठकमा', 'मूल्य', 'वृद्धि', 'बेरोजगारी']),
 ('p11', 'नेपालको जनसंख्या दुई करोड नब्बे लाख पुगेको छ र वृद्धिदर ३.५ प्रतिशत रह्यो।', ['जनसंख्या', 'करोड', 'नब्बे']),
 ('p12', 'सुदूरपश्चिमका पहाडमा बिहानको घाम झुल्किँदा चरा गीत गाउन थाल्छन्।', ['पहाडमा', 'बिहानको', 'चरा']),
 ('p13', 'उनीहरू सँगै बसेर कथा सुन्दै थिए, अनि बुढी हजुरआमाले हाँस्दै भनिन्, बाबु खाऊ।', ['सँगै', 'कथा', 'हजुरआमाले']),
 ('p14', 'राति अबेरसम्म पढेकाले आज बिहान उठ्न गाह्रो भयो।', ['अबेरसम्म', 'बिहान', 'गाह्रो']),
 ('p15', 'विद्यार्थीहरूले परीक्षाको नतिजा हेर्न विश्वविद्यालयको वेबसाइटमा लगइन गरे।', ['विद्यार्थीहरूले', 'परीक्षाको', 'नतिजा']),
 ('p16', 'यो चित्र हेर्दा मलाई मेरो बाल्यकालको सम्झना आयो।', ['चित्र', 'बाल्यकालको', 'सम्झना']),
 ('t01', 'Anthropic ले आज नयाँ मोडेल सार्वजनिक गर्‍यो।', []),
 ('t02', 'Gemini 4 Neon मा Jensen Huang ले GPT-6.2 को तुलना गरे।', []),
 ('t03', 'Liquid Glass Dock नयाँ अपडेटमा सबैभन्दा चर्चित फिचर बन्यो।', []),
 ('t04', 'आजको प्रविधि समाचारमा Anthropic र Gemini को प्रतिस्पर्धा छ।', []),
 ('s01', 'ॐ भूर्भुवः स्वः तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि।', []),
 ('s02', 'सत्यं वद धर्मं चर स्वाध्यायान्मा प्रमदः।', []),
 ('s03', 'असतो मा सद्गमय तमसो मा ज्योतिर्गमय मृत्योर्मा अमृतं गमय।', []),
 ('s04', 'कर्मण्येवाधिकारस्ते मा फलेषु कदाचन।', []),
]
if __name__ == '__main__':
    ck, tag = sys.argv[1], sys.argv[2]
    ns = int(sys.argv[3]) if len(sys.argv) > 3 else 32; gs = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    ref = json.load(open('/work/sp/ref/anchor_ref.json'))
    m = OmniVoice.from_pretrained(ck, device_map='cuda', dtype=torch.bfloat16)
    out = f'/work/sp/evals/{tag}'; os.makedirs(out, exist_ok=True)
    rw, rsr = sf.read('/work/sp/ref/anchor_ref.wav', dtype='float32'); remb = C.spk_emb(rw, rsr)
    res = {}
    for sid, txt, words in S:
        lang = 'sa' if sid.startswith('s') else 'npi'
        tt = txt if lang == 'sa' else nep_norm.normalize(txt)
        torch.manual_seed(1234)
        a = m.generate(text=tt, language=lang, ref_audio='/work/sp/ref/anchor_ref.wav', ref_text=ref['ref_text'], num_step=ns, guidance_scale=gs)
        p = f'{out}/{sid}.wav'; sf.write(p, a[0], m.sampling_rate)
        f = C.score_take(p, txt, remb); hyp = f.get('hyp', '')
        f['cer'] = V.pron_cer(tt, hyp, lang)
        hn = V.norm(hyp); f['word_recall'] = float(np.mean([V.norm(w) in hn for w in words])) if words else None
        f['kind'] = sid[0]; res[sid] = f
    ok = list(res.values()); g = lambda k, f=lambda v: True: float(np.mean([v[k] for v in ok if f(v) and v[k] is not None]))
    summ = dict(tag=tag, n=len(ok), cer_native=g('cer', lambda v: v['kind'] == 'p'), word_recall=g('word_recall'), cer_tech=g('cer', lambda v: v['kind'] == 't'),
                cer_vedic=g('cer', lambda v: v['kind'] == 's'), utmos=g('utmos'), spk_sim=g('spk_sim'), clicks=g('clicks'), hindi=int(sum(v['hindi'] for v in ok)))
    json.dump(dict(summary=summ, takes=res), open(f'{out}/pron_scores.json', 'w'), ensure_ascii=False, indent=1)
    print('PRON', json.dumps({k: round(x, 3) if isinstance(x, float) else x for k, x in summ.items()}), flush=True)
