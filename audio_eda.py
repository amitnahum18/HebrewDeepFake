"""Audio measurements on immutable originals; safe for Windows process workers."""
from pathlib import Path
import hashlib
import av
import numpy as np

def measure_audio(task):
    snapshot, sid=task
    result={'session_id':sid,'decode_error':None}
    try:
        with av.open(str(Path(snapshot)/sid/'audio.mka')) as container:
            stream=container.streams.audio[0]
            rate=stream.codec_context.sample_rate
            result.update(codec=stream.codec_context.name,sample_rate=rate,channels=stream.codec_context.channels)
            convert=av.AudioResampler(format='fltp',layout=stream.codec_context.layout,rate=rate)
            canonical=av.AudioResampler(format='s16',layout='mono',rate=16000)
            chunks=[]; pcmhash=hashlib.sha256()
            for frame in container.decode(stream):
                for cf in canonical.resample(frame):
                    pcmhash.update(cf.to_ndarray().astype('<i2',copy=False).tobytes())
                for ff in convert.resample(frame): chunks.append(ff.to_ndarray())
            for cf in canonical.resample(None):
                pcmhash.update(cf.to_ndarray().astype('<i2',copy=False).tobytes())
        if not chunks: raise ValueError('Empty decoded audio')
        native=np.concatenate(chunks,axis=1); del chunks
        nonfinite=int((~np.isfinite(native)).sum())
        if nonfinite: native=np.nan_to_num(native)
        total=native.size
        peak=float(np.max(np.abs(native)))
        near=int((np.abs(native)>=.999).sum())
        dc=float(native.sum(dtype=np.float64)/total)
        x=native.mean(axis=0,dtype=np.float64).astype(np.float32); del native
        width=max(1,round(rate*.020)); trimmed=x[:len(x)//width*width]
        if len(trimmed):
            frames=trimmed.reshape(-1,width)
            energy=np.sqrt(np.einsum('ij,ij->i',frames,frames,dtype=np.float64)/width)
        else: energy=np.array([np.sqrt(np.einsum('i,i->',x,x,dtype=np.float64)/len(x))])
        db=20*np.log10(np.maximum(energy,1e-12))
        rms=float(np.sqrt(np.einsum('i,i->',x,x,dtype=np.float64)/len(x)))
        result.update(decoded_duration_s=len(x)/rate,peak=peak,rms_dbfs=20*np.log10(max(rms,1e-12)),dc_offset=dc,near_fullscale_fraction=near/total,nonfinite_samples=nonfinite,low_energy_p10_dbfs=float(np.percentile(db,10)),high_energy_p90_dbfs=float(np.percentile(db,90)),energy_dynamic_range_db=float(np.percentile(db,90)-np.percentile(db,10)),crest_factor_db=20*np.log10(max(peak,1e-12)/max(rms,1e-12)),pcm16_16k_mono_sha256=pcmhash.hexdigest(),**{f'quiet_fraction_{abs(th)}dbfs':float((db<th).mean()) for th in [-50,-45,-40]})
    except Exception as e: result['decode_error']=f'{type(e).__name__}: {e}'
    return result
