"""Official SpeechBrain ECAPA layers/weights, isolated CPU inference.

Download weights with --download first; all subsequent inference is offline.
No pretrained classifier or executable remote custom code is loaded.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import pandas as pd
from comparative_audio import read_wav,energy_frames,leading_trailing_quiet

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'voice_distribution'; MODEL=OUT/'model'
MODEL_REVISION='0f99f2d0ebe89ac095bcc5903c4dd8f72b367286'

def download():
    from huggingface_hub import model_info,hf_hub_download
    info=model_info('speechbrain/spkrec-ecapa-voxceleb',revision=MODEL_REVISION)
    MODEL.mkdir(exist_ok=True)
    for filename in ['embedding_model.ckpt','hyperparams.yaml']:
        hf_hub_download('speechbrain/spkrec-ecapa-voxceleb',filename,revision=info.sha,local_dir=MODEL)
    meta=dict(repo='speechbrain/spkrec-ecapa-voxceleb',revision=info.sha,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [MODEL/'embedding_model.ckpt',MODEL/'hyperparams.yaml']})
    (OUT/'model_provenance.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)

def run():
    import torch
    from speechbrain.lobes.features import Fbank
    from speechbrain.processing.features import InputNormalization
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    torch.set_num_threads(4); torch.manual_seed(20261117)
    fbank=Fbank(n_mels=80).eval()
    norm=InputNormalization(norm_type='sentence',std_norm=False).eval()
    model=ECAPA_TDNN(input_size=80,channels=[1024,1024,1024,1024,3072],kernel_sizes=[5,3,3,3,1],dilations=[1,2,3,4,1],attention_channels=128,lin_neurons=192).eval()
    state=torch.load(MODEL/'embedding_model.ckpt',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    frame=pd.read_csv(OUT/'samples.csv')
    paths={r.sample_id:Path(r.absolute_wav_path) for r in frame.itertuples()}
    vectors=[]; mapping=[]; start=time.time()
    output=OUT/'embeddings.npz'
    cached={}
    if output.exists():
        old=np.load(output,allow_pickle=False)
        cached={(str(k),str(v),str(h)):e for k,v,h,e in zip(old['sample_ids'],old['views'],old['wav_hashes'],old['embeddings'])}
    with torch.inference_mode():
        for i,row in enumerate(frame.itertuples()):
            x=read_wav(paths[row.sample_id]).astype(np.float32)
            _,db=energy_frames(x); lead,tail=leading_trailing_quiet(db)
            a=round(min(.25,lead)*16000); b=round(min(.25,tail)*16000)
            views={'original':x,'edge_trimmed':x[a:len(x)-b if b else len(x)]}
            for view,audio in views.items():
                key=(row.sample_id,view,row.wav_sha256)
                if key in cached: embedding=cached[key]
                else:
                    signal=torch.from_numpy(audio.copy()).unsqueeze(0)
                    lengths=torch.ones(1)
                    features=norm(fbank(signal),lengths)
                    embedding=model(features,lengths).squeeze().numpy()
                    embedding=embedding/np.linalg.norm(embedding)
                assert embedding.shape==(192,) and np.isfinite(embedding).all()
                vectors.append(embedding); mapping.append(key)
            if (i+1)%10==0 or i+1==len(frame):
                print('ECAPA',i+1,'/',len(frame),'elapsed_s',round(time.time()-start),flush=True)
                np.savez_compressed(output,embeddings=np.array(vectors),sample_ids=np.array([r[0] for r in mapping]),views=np.array([r[1] for r in mapping]),wav_hashes=np.array([r[2] for r in mapping]))
    import importlib.metadata as metadata
    (OUT/'inference_environment.json').write_text(json.dumps({'versions':{p:metadata.version(p) for p in ['torch','torchaudio','speechbrain','numpy','huggingface-hub']},'device':'cpu','torch_threads':4,'seconds':time.time()-start,'embedding_normalization':'L2 after encode, official encode_batch normalize=False equivalent; sentence mean normalization, no classifier or global embedding normalization','model_state_strict_load':True,'views':['original','edge_trimmed'],'cached_entries_at_start':len(cached)},indent=2),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--download',action='store_true'); args=parser.parse_args()
    download() if args.download else run()
