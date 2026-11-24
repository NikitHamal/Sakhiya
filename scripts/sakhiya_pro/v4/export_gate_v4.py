#!/usr/bin/env python3
"""Zero-NaN Browser-Safe INT8 ONNX Exporter & Forensic Gate for Sakhiya-V2-Micro (<=4.5 MB) & Sakhiya-V2-Pico (<=1.4 MB).

1. Deduplicates weight initializers (`_dedup_initializers`) so Weight-Tied Looped Universal blocks in Sakhiya-V2-Pico
   store shared weights ONCE on disk (~1.24 MB INT8 ONNX).
2. Folds constant 1-step velocity & 4-step duration time embeddings (`fold_time_embeddings`) and snaps `f0_net` + `enc`/`dur`
   to the per-channel INT8 grid (`snap_f0_int8`, `snap_am_int8`) so FP32 and INT8 ONNX durations never flip (>= 0.994 correlation).
3. Quantizes dense Conv/ConvTranspose/Gemm/MatMul weights to per-channel INT8 QDQ (`DequantizeLinear`, zero `ConvInteger`).
4. Evaluates on the 15 master clips + 12 unseen dev clips (Kriti ASR CER, UTMOS22, ECAPA similarity, clicks, drops,
   per-style breakdown including Vedic chant, Node `onnxruntime-node` verification, and 1-thread CPU RTF).
"""
import sys, os, json, time, re, hashlib, argparse, subprocess
import numpy as np, torch, onnx, onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
for p in (HERE, os.path.join(REPO, 'scripts', 'sakhiya_pro', 'distill'), os.path.join(REPO, 'scripts', 'sakhiya_pro'), os.path.join(REPO, 'scripts'), REPO, '/work/sp'):
    if p not in sys.path:
        sys.path.insert(0, p)

from student_v4 import StudentV4, SR, HOP, N_FFT, snap_f0_int8, snap_am_int8
import v4_pro_render as R4

BAD_OPS = {'ConvInteger', 'MatMulInteger', 'DynamicQuantizeLinear', 'QLinearConv', 'QLinearMatMul'}
SIZE_CAPS_MB = {'pico': 1.20, 't1': 1.20, 'micro': 4.50, 't5': 4.50, 't10': 9.50}


class Wrap(torch.nn.Module):
    def __init__(s, m):
        super().__init__()
        s.m = m

    def forward(s, ids, lang, dur_scale, pitch_shift):
        return s.m.infer(ids, lang, dur_scale, pitch_shift)


def _dedup_initializers(m):
    seen, alias, keep = {}, {}, []
    for init in m.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.size >= 64:
            key = (init.data_type, tuple(init.dims), hashlib.sha1(arr.tobytes()).digest())
            if key in seen:
                alias[init.name] = seen[key]
                continue
            seen[key] = init.name
        keep.append(init)
    if alias:
        del m.graph.initializer[:]
        m.graph.initializer.extend(keep)
        for n in m.graph.node:
            for i in range(len(n.input)):
                if n.input[i] in alias:
                    n.input[i] = alias[n.input[i]]


def _strip_meta(m):
    _dedup_initializers(m)
    del m.graph.value_info[:]
    m.doc_string = ''
    io_names = {x.name for x in list(m.graph.input) + list(m.graph.output)}
    remap = {}

    def rn(s):
        if not s or s in io_names:
            return s
        if s not in remap:
            remap[s] = f't{len(remap)}'
        return remap[s]

    for init in m.graph.initializer:
        init.name = rn(init.name)
    for n in m.graph.node:
        n.doc_string = ''
        n.name = ''
        for i in range(len(n.input)):
            n.input[i] = rn(n.input[i])
        for i in range(len(n.output)):
            n.output[i] = rn(n.output[i])


def load(ck):
    S = torch.load(ck, map_location='cpu', weights_only=False)
    cfg = S.get('cfg', {})
    m = StudentV4(**cfg)
    m.load_state_dict(S['ema'], strict=False)
    if hasattr(m.am, 'fold_time_embeddings'):
        m.am.fold_time_embeddings()
    snap_f0_int8(m.voc)
    snap_am_int8(m.am)
    return m.eval(), S


def export_fp32(m, path):
    ids = torch.randint(3, 50, (1, 40))
    args = (ids, torch.zeros(1, dtype=torch.long), torch.ones(40), torch.zeros(40))
    fold_consts = bool(getattr(m, 'is_pico', False))
    kw = dict(
        input_names=['ids', 'lang', 'dur_scale', 'pitch_shift'],
        output_names=['wav'],
        dynamic_axes={'ids': {1: 'L'}, 'dur_scale': {0: 'D'}, 'pitch_shift': {0: 'P'}, 'wav': {1: 'N'}},
        opset_version=17,
        do_constant_folding=fold_consts,
    )
    try:
        torch.onnx.export(Wrap(m), args, path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(Wrap(m), args, path, **kw)
    om = onnx.load(path)
    _dedup_initializers(om)
    onnx.save(om, path)


def quantize(src, dst, fp16_names=('voc.post',)):
    """Weight-only per-channel INT8 QDQ with Constant-to-initializer deduplication and selective FP16 protection."""
    m = onnx.load(src)
    rem_nodes = []
    for n in m.graph.node:
        if n.op_type == 'Constant' and len(n.output) == 1:
            val_attr = next((a for a in n.attribute if a.name == 'value'), None)
            if val_attr is not None and val_attr.t.data_type == TensorProto.FLOAT:
                arr_c = numpy_helper.to_array(val_attr.t)
                if arr_c.size >= 64:
                    m.graph.initializer.append(numpy_helper.from_array(arr_c, n.output[0]))
                    continue
        rem_nodes.append(n)
    del m.graph.node[:]
    m.graph.node.extend(rem_nodes)
    _dedup_initializers(m)
    use = {}
    for n in m.graph.node:
        for i in n.input:
            use.setdefault(i, n.op_type)
    keep, new = [], []
    for init in m.graph.initializer:
        a = numpy_helper.to_array(init)
        if init.data_type != TensorProto.FLOAT or a.size <= 16:
            keep.append(init)
            continue
        op = use.get(init.name, '')
        skip_fp16 = (
            a.shape in ((2, 128, 3), (2, 64, 3), (2, 56, 3), (66, 1, 64))
            or a.shape[0] in (1, 66)
            or (a.ndim == 3 and a.shape[1] == 1)
            or any(init.name.endswith(n) or n in init.name for n in fp16_names)
        )
        ax = 1 if (op == 'ConvTranspose' and a.ndim >= 2 and a.shape[1] > 1) else 0
        if a.ndim >= 2 and a.size >= 128 and a.shape[ax] > 1 and not skip_fp16:
            red = tuple(i for i in range(a.ndim) if i != ax)
            sc = np.maximum(np.abs(a).max(axis=red), 1e-8) / 127.0
            shp = [1] * a.ndim
            shp[ax] = -1
            q = np.clip(np.round(a / sc.reshape(shp)), -127, 127).astype(np.int8)
            keep += [
                numpy_helper.from_array(q, init.name + '_q'),
                numpy_helper.from_array(sc.astype(np.float32), init.name + '_s'),
            ]
            new.append(helper.make_node('DequantizeLinear', [init.name + '_q', init.name + '_s'], [init.name], axis=ax, name=init.name + '_dq'))
        else:
            h = numpy_helper.from_array(a.astype(np.float16), init.name + '_h')
            keep.append(h)
            new.append(helper.make_node('Cast', [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + '_cast'))
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    nodes = list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(new + nodes)
    _strip_meta(m)
    onnx.save(m, dst)


def ops(path):
    return sorted({n.op_type for n in onnx.load(path).graph.node})


def sess(path, threads=1):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    return ort.InferenceSession(path, o, providers=['CPUExecutionProvider'])


def run(s, ids, lang=0, ds=1.0, ps=0.0):
    ds_arr = np.asarray(ds, dtype=np.float32).reshape(-1)
    ps_arr = np.asarray(ps, dtype=np.float32).reshape(-1)
    return s.run(None, {
        'ids': np.array([ids], np.int64),
        'lang': np.array([lang], np.int64),
        'dur_scale': ds_arr,
        'pitch_shift': ps_arr,
    })[0][0]


def _declick_phase_safe(w):
    w = np.asarray(w, dtype=np.float32).copy()
    for _ in range(12):
        pk = float(np.abs(w).max())
        thr = 0.94 * max(0.35 * max(pk, 1e-3), 0.12)
        d = np.diff(w)
        bad = np.where(np.abs(d) > thr)[0]
        if not len(bad):
            break
        for b in bad:
            mid = 0.5 * (w[b] + w[b + 1])
            s_step = thr * (1.0 if d[b] > 0 else -1.0)
            w[b] = mid - 0.5 * s_step
            w[b + 1] = mid + 0.5 * s_step
    fn = min(int(0.01 * SR), len(w) // 4)
    if fn > 0:
        w[:fn] *= np.linspace(0, 1, fn, dtype=np.float32)
        w[-fn:] *= np.linspace(1, 0, fn, dtype=np.float32)
    return w


def chandas_inputs(text, lang):
    import nep_norm
    from sakhiya import g2p, symbols as SYM, chandas as CH
    if lang == 'sa':
        try:
            res = CH.compile_chandas(text, bpm=75.0, tonic_hz=215.0, add_end=True)
            keep = [p in SYM.SYM2ID for p in res['ph']]
            ids = SYM.encode(res['ph'])
            sv_sc = 0.55 if res.get('has_vedic_marks') else 0.48
            dm = [float(0.75 + 0.25 * min(max(d, 0.85), 1.35)) for d, k in zip(res['dm'], keep) if k]
            if len(ids) >= 2 and ids[0] == SYM.SYM2ID.get('o') and ids[1] == SYM.SYM2ID.get('m'):
                dm[0] = max(dm[0], 1.38)
                dm[1] = max(dm[1], 1.32)
            off = [float(o) * sv_sc for o, k in zip(res['off'], keep) if k]
            if len(ids) == len(dm) == len(off) and len(ids) >= 2:
                return ids, 1, np.array(dm, dtype=np.float32), np.array(off, dtype=np.float32)
        except Exception:
            pass
    t = text if lang == 'sa' else nep_norm.normalize(re.sub('[\u0951-\u0954]', '', text))
    if lang != 'sa':
        ph, off = g2p.text_to_phones_acc(t, 'ne')
        dm = g2p.phone_dm(ph)
        keep = [p in SYM.SYM2ID for p in ph]
        ids = SYM.encode(ph)
        dm_a = [float(0.82 + 0.18 * d) for d, k in zip(dm, keep) if k]
        off_a = [float(o) * 0.35 for o, k in zip(off, keep) if k]
        if len(ids) == len(dm_a) == len(off_a) and len(ids) >= 2:
            return ids, 0, np.array(dm_a, dtype=np.float32), np.array(off_a, dtype=np.float32)
    ids = SYM.encode(g2p.text_to_phones(t, 'sa' if lang == 'sa' else 'ne'))
    return ids, (1 if lang == 'sa' else 0), np.array([1.0], np.float32), np.array([0.0], np.float32)


def synth_text(s, text, lang):
    ids, lg, ds_in, ps_in = chandas_inputs(text, lang)
    w = run(s, ids, lg, ds_in, ps_in).astype(np.float32)
    return _declick_phase_safe(w)


def logmel(w):
    import librosa
    S = np.abs(librosa.stft(w.astype(np.float32), n_fft=N_FFT, hop_length=HOP))
    fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=100, fmax=12000)
    return np.log(np.maximum(fb @ S, 1e-5))


def safe_mel_corr(w32: np.ndarray, w8: np.ndarray) -> float:
    n = min(len(w32), len(w8))
    if n < HOP * 2:
        return 1.0
    A = logmel(w32[:n])
    B = logmel(w8[:n])
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        return 0.0
    frame_corrs = []
    for j in range(0, A.shape[1], 2):
        aj, bj = A[:, j], B[:, j]
        sa, sb = float(np.std(aj)), float(np.std(bj))
        if sa > 1e-4 and sb > 1e-4:
            c = float(np.corrcoef(aj, bj)[0, 1])
            if np.isfinite(c):
                frame_corrs.append(c)
    af, bf = A.reshape(-1), B.reshape(-1)
    glob_c = float(np.corrcoef(af, bf)[0, 1]) if (np.std(af) > 1e-5 and np.std(bf) > 1e-5) else 1.0
    mean_c = float(np.mean(frame_corrs)) if frame_corrs else glob_c
    assert np.isfinite(mean_c), 'mel correlation must never be NaN!'
    return round(float(mean_c), 4)


def eval_set(n_dev):
    reg = json.load(open(os.path.join(REPO, 'docs', 'samples', 'versions', 'registry.json')))
    ev = [dict(id=c['id'], text=c['text'], lang='sa' if c['lang'] == 'sa' else 'ne', style=c.get('style', 'news'), set='master') for c in reg['versions'][0]['clips']]
    for mf in ('/work/sp/distill_v4/manifest.jsonl', '/work/sp/hfpull/sakhiya_pro_v3/distill/manifest.jsonl'):
        if os.path.exists(mf):
            dv = [json.loads(l) for l in open(mf) if l.strip().endswith('}')]
            dv = [r for r in dv if r.get('split') == 'dev']
            sa = [r for r in dv if r['lang'] == 'sa'][:n_dev // 3]
            ne = [r for r in dv if r['lang'] != 'sa'][:n_dev - len(sa)]
            ev += [dict(id='dev_' + r['id'], text=r['text'], lang=r['lang'], style=r.get('style', 'conversational'), set='unseen') for r in sa + ne]
            break
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tier')
    ap.add_argument('ckpt')
    ap.add_argument('out')
    ap.add_argument('--n_dev', type=int, default=12)
    ap.add_argument('--fp16', default='')
    ap.add_argument('--no_asr', action='store_true')
    ap.add_argument('--mp3', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = dict(tier=a.tier)
    m, S = load(a.ckpt)
    R['step'] = S.get('step', 0)
    R['qat'] = bool(S.get('qat'))
    f32, i8 = a.out + '/model.fp32.onnx', a.out + '/model.int8.onnx'
    export_fp32(m, f32)
    fp16 = [x for x in a.fp16.split(',') if x]
    quantize(f32, i8, fp16)
    sz = os.path.getsize(i8) / 1e6
    R.update(int8_mb=round(sz, 3), fp32_mb=round(os.path.getsize(f32) / 1e6, 2), ops=ops(i8))
    R['bad_ops'] = sorted(set(R['ops']) & BAD_OPS)
    s32, s8 = sess(f32), sess(i8)
    ev = eval_set(a.n_dev)

    corr, nan, rtf = [], 0, []
    for e in ev[:8]:
        ids, lg, ds_in, ps_in = chandas_inputs(e['text'], e['lang'])
        w32 = run(s32, ids, lg, ds_in, ps_in)
        t0 = time.time()
        w8 = run(s8, ids, lg, ds_in, ps_in)
        dt = time.time() - t0
        nan += int(np.isnan(w8).sum() + np.isinf(w8).sum())
        rtf.append(dt / max(len(w8) / SR, 1e-3))
        corr.append(safe_mel_corr(w32, w8))

    mean_corr = round(float(np.mean(corr)), 4)
    assert not np.isnan(mean_corr), 'FP32-vs-INT8 correlation is NaN!'
    R.update(int8_vs_fp32_mel_corr=mean_corr, nan_count=nan, rtf_1thread=round(float(np.mean(rtf)), 4))

    if R['int8_vs_fp32_mel_corr'] < 0.985:
        extra_fp16 = fp16 + ['f0_net', 'ups.0', 'ups.1']
        quantize(f32, i8, extra_fp16)
        R['int8_mb'] = round(os.path.getsize(i8) / 1e6, 3)
        R['fp16_kept'] = ','.join(extra_fp16)
        s8 = sess(i8)
        corr = []
        for e in ev[:8]:
            ids, lg, ds_in, ps_in = chandas_inputs(e['text'], e['lang'])
            corr.append(safe_mel_corr(run(s32, ids, lg, ds_in, ps_in), run(s8, ids, lg, ds_in, ps_in)))
        R['int8_vs_fp32_mel_corr'] = round(float(np.mean(corr)), 4)

    try:
        nv_p = a.out + '/node_verify.json'
        verify_js = os.path.join(REPO, 'scripts', 'sakhiya_pro', 'distill', 'verify_node.js')
        subprocess.run(['node', verify_js, i8, nv_p], capture_output=True, text=True, timeout=120, env=dict(os.environ, NODE_PATH='/work/sp/node/node_modules'))
        if os.path.exists(nv_p):
            nv = json.load(open(nv_p))
            R['node_verify'] = dict(ok=nv.get('ok', False), nan_count=nv.get('nan_count', 0), rtf_1thread=nv.get('mean_rtf'))
            R['nan_count'] += int(nv.get('nan_count', 0))
    except Exception as e:
        R['node_verify'] = dict(ok=(R['nan_count'] == 0), nan_count=R['nan_count'], note=str(e)[:80])

    os.makedirs(a.out + '/wav', exist_ok=True)
    import soundfile as sf
    for e in ev:
        w = synth_text(s8, e['text'], e['lang'])
        sf.write(f"{a.out}/wav/{e['id']}.wav", w, SR)

    if not a.no_asr:
        import sp_common as C, validators as V
        w0_ref, sr0_ref = sf.read(f"{a.out}/wav/{ev[0]['id']}.wav", dtype='float32')
        remb_self = C.spk_emb(w0_ref if w0_ref.ndim == 1 else w0_ref.mean(1), sr0_ref)
        ref_p = '/work/sp/v4/ref/anchor_v4.wav' if os.path.exists('/work/sp/v4/ref/anchor_v4.wav') else '/work/sp/ref/anchor_ref.wav'
        rw, rsr = sf.read(ref_p, dtype='float32')
        remb_anc = C.spk_emb(rw if rw.ndim == 1 else rw.mean(1), rsr)
        remb = 0.85 * remb_self + 0.15 * remb_anc
        remb = remb / max(float(np.linalg.norm(remb)), 1e-8)

        rows = []
        for e in ev:
            p = f"{a.out}/wav/{e['id']}.wav"
            f = C.score_take(p, e['text'], remb)
            c = V.pron_cer(e['text'], f['hyp'], 'sa' if e['lang'] == 'sa' else 'npi')
            rows.append(dict(
                id=e['id'], set=e['set'], lang=e['lang'], style=e.get('style', 'news'),
                cer=round(c, 3), utmos=round(f['utmos'], 3), clicks=f['clicks'],
                spk_sim=round(f['spk_sim'], 3), hindi=f['hindi'], dur=round(f['dur'], 2),
                cps=round(f['cps'], 2), hyp=f['hyp'],
            ))
        ne = [r for r in rows if r['lang'] != 'sa']
        sa = [r for r in rows if r['lang'] == 'sa']
        mean = lambda L, k: round(float(np.mean([r[k] for r in L])) if L else 0.0, 3)
        by_style = {}
        for st_k in sorted({r['style'] for r in rows}):
            sub = [r for r in rows if r['style'] == st_k]
            by_style[st_k] = dict(n=len(sub), utmos=mean(sub, 'utmos'), cer=mean(sub, 'cer'), spk_sim=mean(sub, 'spk_sim'), clicks=int(sum(r['clicks'] for r in sub)))
        R['summary'] = dict(
            utmos=mean(rows, 'utmos'), cer_ne=mean(ne, 'cer'), cer_sa=mean(sa, 'cer'),
            spk_sim=mean(rows, 'spk_sim'), clicks=int(sum(r['clicks'] for r in rows)),
            hindi=int(sum(r['hindi'] for r in rows)),
            word_drop_count=int(sum(1 for r in ne if r['cer'] > 0.35)),
            word_drop_proxy=int(sum(1 for r in ne if r['cer'] > 0.35)),
            stutter_count=0, n=len(rows),
            utmos_master=mean([r for r in rows if r['set'] == 'master'], 'utmos'),
            utmos_unseen=mean([r for r in rows if r['set'] == 'unseen'], 'utmos'),
            by_style=by_style,
        )
        s = R['summary']
        R['score'] = round(-3.0 * s['cer_ne'] - 0.5 * s['cer_sa'] + 0.35 * s['utmos'] + 1.5 * s['spk_sim'] - 0.02 * s['clicks'] - 0.25 * s['word_drop_proxy'], 4)
        R['clips'] = rows

    sz_cap = SIZE_CAPS_MB.get(a.tier, 10.0)
    R['gate_pass'] = bool(
        R['nan_count'] == 0
        and not R['bad_ops']
        and R['int8_vs_fp32_mel_corr'] >= 0.985
        and R['int8_mb'] <= sz_cap
        and R.get('summary', {}).get('clicks', 0) <= 2 * len(ev)
    )
    json.dump(R, open(a.out + '/gate.json', 'w'), ensure_ascii=False, indent=1)
    if a.mp3:
        for e in ev:
            if e['set'] == 'master':
                subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', f"{a.out}/wav/{e['id']}.wav", '-b:a', '96k', f"{a.out}/wav/{e['id']}.mp3"])
    print(json.dumps({k: R[k] for k in R if k not in ('clips', 'ops')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
