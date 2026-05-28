import numpy as np
from obspy import Stream
from .normalize import rms_normalize, onebit_normalize
from .whiten import spectral_whiten


def preprocess_stream(stream: Stream,
                      bandpass: tuple[float, float],
                      normalization: str,
                      whiten: bool) -> Stream:
    """Detrend → bandpass → normalize → whiten."""
    st = stream.copy()
    st.detrend("linear")
    st.taper(max_percentage=0.05)
    st.filter("bandpass", freqmin=bandpass[0], freqmax=bandpass[1], corners=4)

    for tr in st:
        fs = tr.stats.sampling_rate
        data = tr.data.astype(np.float64)

        if normalization == "rms":
            data = rms_normalize(data, win_len=int(fs))
        elif normalization == "1bit":
            data = onebit_normalize(data)

        if whiten:
            data = spectral_whiten(data, fs, freq_range=bandpass)

        tr.data = data.astype(np.float32)

    return st
