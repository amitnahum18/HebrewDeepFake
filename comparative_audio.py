"""Deterministic, identical acoustic measurements for saved real/synthetic WAVs."""
from pathlib import Path
import wave
import numpy as np


def read_wav(path):
    with wave.open(str(path),'rb') as stream:
        if stream.getnchannels()!=1 or stream.getframerate()!=16000 or stream.getsampwidth()!=2:
            raise ValueError('Expected 16 kHz mono PCM16 WAV')
        return np.frombuffer(stream.readframes(stream.getnframes()),dtype='<i2').astype(np.float64)/32768


def energy_frames(x):
    width=320
    if len(x)<width: return np.empty((0,width)),np.array([])
    frames=x[:len(x)//width*width].reshape(-1,width)
    db=20*np.log10(np.maximum(np.sqrt(np.mean(frames*frames,axis=1)),1e-12))
    return frames,db


def leading_trailing_quiet(db,threshold=-45):
    active=np.flatnonzero(db>=threshold)
    if not len(active): return len(db)*.02,len(db)*.02
    return active[0]*.02,(len(db)-1-active[-1])*.02


def features(x):
    frames,db=energy_frames(x)
    if not len(frames): raise ValueError('Too short for 20 ms frame statistics')
    rms=float(np.sqrt(np.mean(x*x))); peak=float(np.max(np.abs(x)))
    lead,tail=leading_trailing_quiet(db)
    # Spectral statistics use only frames above the absolute energy threshold.
    active=frames[db>=-45]
    if len(active):
        power=np.abs(np.fft.rfft(active*np.hanning(320),axis=1))**2
        frequency=np.fft.rfftfreq(320,1/16000)
        denominator=np.maximum(power.sum(axis=1),1e-20)
        centroid=float(np.median((power*frequency).sum(axis=1)/denominator))
        highband=float(np.median(power[:,frequency>=4000].sum(axis=1)/denominator))
        flatness=float(np.median(np.exp(np.mean(np.log(np.maximum(power,1e-20)),axis=1))/np.maximum(power.mean(axis=1),1e-20)))
    else: centroid=highband=flatness=np.nan
    return {'duration_s':len(x)/16000,'rms_dbfs':20*np.log10(max(rms,1e-12)),'peak':peak,
            'dc_offset':float(x.mean()),'near_fullscale_fraction':float((np.abs(x)>=.999).mean()),
            'quiet_fraction_50dbfs':float((db< -50).mean()),'quiet_fraction_45dbfs':float((db< -45).mean()),
            'quiet_fraction_40dbfs':float((db< -40).mean()),'leading_quiet_s':lead,'trailing_quiet_s':tail,
            'edge_quiet_fraction':min(1,(lead+tail)/(len(x)/16000)),
            'energy_dynamic_range_db':float(np.percentile(db,90)-np.percentile(db,10)),
            'zero_crossing_fraction':float(np.mean(np.signbit(x[1:])!=np.signbit(x[:-1]))),
            'spectral_centroid_hz':centroid,'highband_energy_fraction':highband,'spectral_flatness':flatness}


def analyze_one(row):
    path=Path(row['absolute_wav_path']); x=read_wav(path)
    result={**row,**features(x)}
    lead,tail=result['leading_quiet_s'],result['trailing_quiet_s']
    trim_start=round(min(.25,lead)*16000); trim_end=round(min(.25,tail)*16000)
    trimmed=x[trim_start:len(x)-trim_end if trim_end else len(x)]
    result['analysis_edge_removed_s']=(len(x)-len(trimmed))/16000
    result.update({'edge_trimmed_'+k:v for k,v in features(trimmed).items()})
    rms=np.sqrt(np.mean(x*x)); target=10**(-23/20)
    gain=min(target/max(rms,1e-12),.99/max(result['peak'],1e-12))
    result['level_normalization_gain_db']=20*np.log10(max(gain,1e-12))
    result.update({'level_normalized_'+k:v for k,v in features(x*gain).items()})
    return result
