from pathlib import Path
from obspy import read, Stream
from obspy.core import UTCDateTime


def read_sac_segments(sac_dir: str, segment_length: int) -> list[tuple[str, Stream]]:
    """
    Read all SAC files in sac_dir, merge by channel, split into
    fixed-length segments. Returns list of (segment_id, Stream).
    segment_id is ISO-8601 string of segment start time.
    """
    sac_dir = Path(sac_dir)
    files = sorted(sac_dir.rglob("*.sac")) + sorted(sac_dir.rglob("*.SAC"))
    if not files:
        return []

    st = Stream()
    for f in files:
        st += read(str(f))
    st.merge(fill_value=0)
    st.sort()

    start = min(tr.stats.starttime for tr in st)
    end   = max(tr.stats.endtime   for tr in st)

    segments: list[tuple[str, Stream]] = []
    t = start
    while t < end:
        t_end = t + segment_length
        seg = st.slice(t, t_end, nearest_sample=False)
        if seg:
            segments.append((str(t), seg))
        t = t_end

    return segments
