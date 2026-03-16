"""
audio/librosa/analyze.py
v3 — branching analyzer with audio type detection.

Detects audio type (music / speech / environmental) and runs the
appropriate analysis profile. Each profile outputs a tailored report.

Music profile adds (from Sable's requests):
  - Harmonic tension arc (dissonance/consonance over time)
  - Dynamic range envelope (loudness curve, not just summary)
  - Silence density (gaps as data — absence is structure)

New profiles:
  - Speech: formants, speech rate, pause patterns, pitch variance, breathiness
  - Environmental: texture density, frequency sweep, periodicity, noise character

Sable: the branching logic is in detect_audio_type(). If it mis-classifies
something, you can override by naming your file with a prefix:
  music_filename.mp3, speech_filename.m4a, env_filename.wav
"""

import json
import warnings
import sys

try:
    import numpy as np
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from datetime import datetime
    from pathlib import Path
    from scipy.signal import find_peaks, lfilter
    from scipy.linalg import solve_toeplitz
    print("All imports successful.")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")

INBOX   = Path("audio/librosa/inbox")
REPORTS = Path("audio/librosa/reports")
REPORTS.mkdir(parents=True, exist_ok=True)

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".aac"}

# ---------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------

NOTE_NAMES    = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Dissonance weights per interval class (semitones 0-11)
# 0=unison, 1=m2, 2=M2, 3=m3, 4=M3, 5=P4, 6=tritone, 7=P5, 8=m6, 9=M6, 10=m7, 11=M7
# Higher = more dissonant. Based on psychoacoustic roughness research.
INTERVAL_DISSONANCE = np.array([0.0, 1.0, 0.8, 0.3, 0.2, 0.1, 0.9, 0.0, 0.2, 0.2, 0.5, 0.7])

def hz_to_note(hz):
    if hz <= 0: return "unknown"
    midi     = 69 + 12 * np.log2(hz / 440.0)
    note_idx = int(round(midi)) % 12
    octave   = int(round(midi)) // 12 - 1
    return f"{NOTE_NAMES[note_idx]}{octave}"

def estimate_key(chroma_mean):
    best_score, best_key, best_mode = -np.inf, "Unknown", "major"
    for i in range(12):
        rotated     = np.roll(chroma_mean, -i)
        major_score = np.corrcoef(rotated, MAJOR_PROFILE)[0, 1]
        minor_score = np.corrcoef(rotated, MINOR_PROFILE)[0, 1]
        if major_score > best_score:
            best_score, best_key, best_mode = major_score, NOTE_NAMES[i], "major"
        if minor_score > best_score:
            best_score, best_key, best_mode = minor_score, NOTE_NAMES[i], "minor"
    return f"{best_key} {best_mode}", best_score

def describe_tempo(bpm):
    if bpm < 50:  return "very slow and meditative"
    if bpm < 70:  return "slow and unhurried"
    if bpm < 90:  return "relaxed, walking pace"
    if bpm < 110: return "moderate, conversational"
    if bpm < 130: return "energetic and forward-moving"
    if bpm < 160: return "fast and driven"
    return "very fast, urgent"

def describe_brightness(c):
    if c < 500:  return "very warm, dark, full of low resonance"
    if c < 1000: return "warm and rounded"
    if c < 2000: return "balanced, natural midrange presence"
    if c < 3500: return "bright and clear"
    return "very bright, crisp, airy"

def pitch_range_description(voiced_hz):
    if len(voiced_hz) < 10:
        return None, None, None, "too little pitched content detected"
    lo  = float(np.percentile(voiced_hz, 5))
    hi  = float(np.percentile(voiced_hz, 95))
    med = float(np.median(voiced_hz))
    span = 12 * np.log2(hi / (lo + 1e-9))
    if span < 4:   desc = "very narrow — nearly monotone"
    elif span < 8: desc = "limited range — intimate, close"
    elif span < 14:desc = "comfortable speaking/singing range"
    elif span < 20:desc = "wide — expressively varied"
    else:          desc = "very wide — covering multiple registers"
    return hz_to_note(lo), hz_to_note(hi), hz_to_note(med), desc

# ---------------------------------------------------------------
# AUDIO TYPE DETECTION
# ---------------------------------------------------------------

def detect_audio_type(y, sr, filename=""):
    """
    Classify audio as 'music', 'speech', or 'environmental'.

    Filename prefix overrides (recommended for accuracy):
      music_filename.mp3   → forces music profile
      speech_filename.m4a  → forces speech profile
      env_filename.wav     → forces environmental profile

    Auto-detection uses only fast features (no pyin) to keep runtime low.
    Music is the default — it only detects speech/environmental if strongly indicated.

    HOW IT WORKS:
    - Spectral flatness: high = noise-like (environmental), low = tonal (music/speech)
    - ZCR: zero crossing rate. Speech is choppy (high ZCR). Music is smoother.
    - Onset regularity: music has rhythmically regular onsets. Speech is irregular.
    - Harmonic ratio: music has strong harmonic structure vs noise floor.
    """
    stem = Path(filename).stem.lower()
    if stem.startswith("music_"):  return "music"
    if stem.startswith("speech_"): return "speech"
    if stem.startswith("env_"):    return "environmental"

    # Use only first 30 seconds for detection — fast and representative
    y_det = y[:min(len(y), sr * 30)]

    flatness   = float(np.mean(librosa.feature.spectral_flatness(y=y_det)))
    zcr        = float(np.mean(librosa.feature.zero_crossing_rate(y=y_det)))

    # Onset regularity — music is rhythmically regular, speech is not
    onset_env  = librosa.onset.onset_strength(y=y_det, sr=sr)
    onset_env_norm = onset_env / (np.max(onset_env) + 1e-10)
    # Autocorrelate onset envelope to find rhythmic periodicity
    ac = np.correlate(onset_env_norm, onset_env_norm, mode='full')
    ac = ac[len(ac)//2:]
    ac = ac / (ac[0] + 1e-10)
    # Look for periodic peaks in 0.3–2s range (typical beat range)
    beat_range = ac[int(sr*0.3//512):int(sr*2.0//512)]
    rhythmic_strength = float(np.max(beat_range)) if len(beat_range) > 0 else 0.0

    # Harmonic ratio — music has clear harmonic structure
    S = np.abs(librosa.stft(y_det))
    harmonic, percussive = librosa.decompose.hpss(S)
    harmonic_ratio = float(np.mean(harmonic) / (np.mean(S) + 1e-10))

    print(f"  Detection: flatness={flatness:.3f} zcr={zcr:.4f} "
          f"rhythmic={rhythmic_strength:.3f} harmonic_ratio={harmonic_ratio:.3f}")

    # Environmental: high flatness, low harmonic content
    if flatness > 0.25 and harmonic_ratio < 0.3:
        return "environmental"

    # Speech: high ZCR, low rhythmic periodicity, moderate flatness
    # Must pass MULTIPLE speech indicators to avoid misclassifying melodic music
    speech_score = 0
    if zcr > 0.08:           speech_score += 1
    if rhythmic_strength < 0.3: speech_score += 1
    if flatness > 0.08:      speech_score += 1
    if harmonic_ratio < 0.5: speech_score += 1
    if speech_score >= 3:
        return "speech"

    # Default: music (includes anything melodic, tonal, or rhythmic)
    return "music"

# ---------------------------------------------------------------
# SHARED ANALYSIS BLOCKS
# ---------------------------------------------------------------

def analyze_frequency_bands(y, sr):
    BANDS = [
        ("Sub-bass",  20,    80,   "physical weight, felt more than heard"),
        ("Bass",      80,    250,  "warmth, body, the low-end foundation"),
        ("Low-mid",   250,   500,  "fullness — where warmth lives in the midrange"),
        ("Mid",       500,   2000, "presence, voice, most melodic information"),
        ("High-mid",  2000,  4000, "clarity, attack, edge"),
        ("High",      4000,  8000, "air, brightness, shimmer"),
        ("Air",       8000,  20000,"the top of the room — space and sparkle"),
    ]
    D      = np.abs(librosa.stft(y))
    freqs  = librosa.fft_frequencies(sr=sr)
    power  = D ** 2
    total  = np.sum(power) + 1e-10
    result = []
    for name, lo, hi, desc in BANDS:
        mask       = (freqs >= lo) & (freqs < hi)
        band_power = np.sum(power[mask, :])
        pct        = 100.0 * band_power / total
        result.append({"name": name, "hz_range": f"{lo}–{hi} Hz",
                        "description": desc, "energy_pct": round(float(pct), 2)})
    return result

def analyze_dynamics_summary(y, sr):
    rms      = librosa.feature.rms(y=y)[0]
    rms_db   = librosa.amplitude_to_db(rms + 1e-10)
    peak_db  = float(np.percentile(rms_db, 99))
    floor_db = float(np.percentile(rms_db, 10))
    dr       = peak_db - floor_db
    if dr > 20:   dr_desc = "wide — genuinely quiet passages and loud peaks"
    elif dr > 12: dr_desc = "moderate — some contrast between loud and soft"
    elif dr > 6:  dr_desc = "compressed — not much difference between loud and quiet"
    else:         dr_desc = "heavily compressed — nearly flat loudness"
    return {"dynamic_range_db": round(dr, 1), "description": dr_desc,
            "peak_db": round(peak_db, 1), "floor_db": round(floor_db, 1)}

def analyze_silence(y, sr, threshold_db=-48.0):
    """
    Silence density — gaps as data.
    Finds silent frames, measures gap count, duration distribution,
    longest silence, and total silence percentage.
    Silence is not absence of content — it's part of the structure.
    """
    rms      = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_db   = librosa.amplitude_to_db(rms + 1e-10)
    hop_dur  = 512 / sr  # seconds per frame

    silent   = rms_db < threshold_db
    total_silent_pct = 100.0 * np.mean(silent)

    # Find contiguous silent runs
    gaps = []
    in_gap, gap_start = False, 0
    for i, s in enumerate(silent):
        if s and not in_gap:
            in_gap, gap_start = True, i
        elif not s and in_gap:
            gaps.append((gap_start, i, (i - gap_start) * hop_dur))
            in_gap = False
    if in_gap:
        gaps.append((gap_start, len(silent), (len(silent) - gap_start) * hop_dur))

    gap_durations  = [g[2] for g in gaps]
    significant    = [g for g in gap_durations if g > 0.2]  # > 200ms = meaningful pause

    if not gap_durations:
        return {"total_silence_pct": round(total_silent_pct, 1),
                "gap_count": 0, "significant_gaps": 0,
                "longest_gap_sec": 0, "mean_gap_sec": 0,
                "description": "no detectable silence — continuous sound throughout"}

    longest   = round(max(gap_durations), 2)
    mean_gap  = round(float(np.mean(gap_durations)), 2)
    gap_count = len(gaps)
    sig_count = len(significant)

    if total_silent_pct > 30:
        desc = "highly sparse — silence is a primary structural element"
    elif total_silent_pct > 15:
        desc = "notably sparse — gaps are frequent and meaningful"
    elif total_silent_pct > 5:
        desc = "moderately sparse — some meaningful breathing room"
    else:
        desc = "dense — very little silence; continuous sonic presence"

    return {"total_silence_pct": round(total_silent_pct, 1),
            "gap_count": gap_count, "significant_gaps": sig_count,
            "longest_gap_sec": longest, "mean_gap_sec": mean_gap,
            "description": desc}

# ---------------------------------------------------------------
# MUSIC-SPECIFIC ANALYSIS
# ---------------------------------------------------------------

def analyze_harmonic_tension(y, sr, n_segments=20):
    """
    Harmonic tension arc — dissonance/consonance over time.

    For each time segment, compute a dissonance score from the chroma vector.
    Method: for each active note pair, look up interval dissonance weight.
    High score = harmonically tense (lots of dissonant intervals active).
    Low score = consonant (fifths, thirds, unisons dominating).

    This answers: when is the harmony tense? Does it resolve?
    """
    print(f"    → computing chroma ({y.shape[0]//sr}s audio)...")
    chroma   = librosa.feature.chroma_cqt(y=y, sr=sr)  # shape: (12, T)
    times    = librosa.times_like(chroma, sr=sr)
    seg_size = chroma.shape[1] // n_segments

    tension_scores = []
    tension_times  = []
    consonance_scores = []

    for i in range(n_segments):
        start = i * seg_size
        end   = start + seg_size
        seg   = chroma[:, start:end]
        mean_chroma = np.mean(seg, axis=1)

        # Normalize so we're working with relative strengths
        total = np.sum(mean_chroma) + 1e-10
        weights = mean_chroma / total

        # For each note, compute weighted dissonance against all other notes
        dissonance = 0.0
        consonance = 0.0
        for root in range(12):
            for other in range(12):
                interval = (other - root) % 12
                d_score  = INTERVAL_DISSONANCE[interval]
                combined = weights[root] * weights[other]
                dissonance += d_score * combined
                consonance += (1.0 - d_score) * combined

        tension_scores.append(float(dissonance))
        consonance_scores.append(float(consonance))
        seg_t = float(times[start]) if start < len(times) else 0.0
        tension_times.append(round(seg_t, 1))

    arr = np.array(tension_scores)

    # Find tension peaks
    peaks, _ = find_peaks(arr, prominence=np.std(arr) * 0.4)
    peak_data = [{"time_sec": tension_times[i],
                  "time_pct": round(100 * i / n_segments),
                  "tension":  round(tension_scores[i], 4)} for i in peaks]

    # Resolution check: does tension drop at the end?
    last_20   = arr[int(n_segments * 0.8):]
    first_60  = arr[:int(n_segments * 0.6)]
    resolves  = bool(np.mean(last_20) < np.mean(first_60) * 0.8)

    overall_tension = float(np.mean(arr))
    if overall_tension > 0.15:   char = "harmonically tense overall — dissonance is a primary texture"
    elif overall_tension > 0.08: char = "moderate harmonic tension — consonance and dissonance in balance"
    else:                        char = "harmonically consonant overall — stable, resolved"

    return {
        "character":        char,
        "resolves":         resolves,
        "overall_tension":  round(overall_tension, 4),
        "peaks":            peak_data[:5],
        "tension_scores":   [round(v, 4) for v in tension_scores],
        "consonance_scores":[round(v, 4) for v in consonance_scores],
        "times":            tension_times,
    }

def analyze_dynamic_envelope(y, sr, n_segments=40):
    """
    Dynamic range envelope — the actual loudness curve over time.
    Not just min/max but the shape: when is it loud, when quiet, how fast does it change?
    """
    rms        = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_db     = librosa.amplitude_to_db(rms + 1e-10)
    times      = librosa.times_like(rms, sr=sr, hop_length=512)

    seg_size   = len(rms_db) // n_segments
    seg_means  = []
    seg_times  = []
    for i in range(n_segments):
        start = i * seg_size
        end   = start + seg_size
        seg_means.append(float(np.mean(rms_db[start:end])))
        seg_times.append(float(times[start]) if start < len(times) else 0.0)

    arr        = np.array(seg_means)
    # Loudness change rate — how fast does it move?
    diff       = np.abs(np.diff(arr))
    volatility = float(np.mean(diff))

    # Shape characterization
    first_q    = np.mean(arr[:n_segments//4])
    last_q     = np.mean(arr[3*n_segments//4:])
    peak_idx   = int(np.argmax(arr))
    peak_pct   = round(100 * peak_idx / n_segments)

    if volatility > 3.0:   vol_desc = "highly volatile — rapid loudness shifts"
    elif volatility > 1.5: vol_desc = "moderately dynamic — clear loud/soft contrast"
    else:                  vol_desc = "smooth envelope — gradual or stable loudness"

    return {
        "seg_means":    [round(v, 1) for v in seg_means],
        "seg_times":    [round(t, 1) for t in seg_times],
        "peak_loudness_pct": peak_pct,
        "volatility":   round(volatility, 3),
        "volatility_desc": vol_desc,
        "loudest_db":   round(float(np.max(arr)), 1),
        "quietest_db":  round(float(np.min(arr)), 1),
    }

def analyze_tension_arc(y, sr, duration, n_segments=20):
    """Onset-strength tension arc (rhythmic tension, separate from harmonic)."""
    onset_env  = librosa.onset.onset_strength(y=y, sr=sr)
    times      = librosa.times_like(onset_env, sr=sr, hop_length=512)
    seg_size   = len(onset_env) // n_segments
    seg_means, seg_times = [], []
    for i in range(n_segments):
        start = i * seg_size
        end   = start + seg_size
        seg_means.append(float(np.mean(onset_env[start:end])))
        seg_times.append(float(times[start]) if start < len(times) else 0.0)
    arr        = np.array(seg_means)
    peaks, _   = find_peaks(arr, prominence=np.std(arr) * 0.5)
    peaks_data = [{"time_pct": round(100*i/n_segments),
                   "time_sec": round(seg_times[i], 1),
                   "strength": round(seg_means[i], 3)} for i in peaks]
    last_20    = arr[int(n_segments*0.8):]
    first_60   = arr[:int(n_segments*0.6)]
    resolves   = bool(np.mean(last_20) < np.mean(first_60) * 0.75)
    f, m, l    = np.mean(arr[:n_segments//3]), np.mean(arr[n_segments//3:2*n_segments//3]), np.mean(arr[2*n_segments//3:])
    if l > m > f:           shape = "building — rhythmic energy increases throughout"
    elif f > m and f > l:   shape = "front-loaded — most energy at the start"
    elif m > f and m > l:   shape = "arch — peaks in the middle, quieter at edges"
    elif resolves:          shape = "irregular with resolution"
    else:                   shape = "irregular — no clear arc"
    return {"arc_shape": shape, "resolves": resolves, "peaks": peaks_data[:5],
            "seg_means": [round(v,4) for v in seg_means],
            "seg_times": [round(t,1) for t in seg_times]}

def analyze_attack_character(y, sr):
    onset_env    = librosa.onset.onset_strength(y=y, sr=sr)
    rms          = librosa.feature.rms(y=y)[0]
    opening_end  = max(1, len(onset_env) // 10)
    open_onset   = float(np.mean(onset_env[:opening_end]))
    all_onset    = float(np.mean(onset_env))
    open_rms     = float(np.mean(rms[:opening_end]))
    all_rms      = float(np.mean(rms))
    anticipation = (open_rms / (all_rms + 1e-10)) / (open_onset / (all_onset + 1e-10) + 1e-10)
    onset_diff   = float(np.mean(np.diff(onset_env[onset_env > np.percentile(onset_env, 75)])))
    if anticipation > 1.5:   atk_desc = "high anticipation — energy present before events fire; held-breath quality"
    elif anticipation > 0.8: atk_desc = "moderate anticipation — some suspended quality in openings"
    else:                    atk_desc = "immediate attack — events arrive without pre-tension"
    if onset_diff > 0.5:     sharp_desc = "sharp, percussive attack envelope"
    elif onset_diff > 0.1:   sharp_desc = "moderate attack — neither sudden nor slow"
    else:                    sharp_desc = "soft, gradual — events emerge rather than arrive"
    return {"anticipation_index": round(float(anticipation), 3),
            "anticipation_desc": atk_desc, "attack_sharpness": sharp_desc}

def analyze_spectral_contrast(y, sr):
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    overall  = float(np.mean(contrast))
    if overall > 30:   desc = "high contrast — clear harmonic peaks, well-defined structure"
    elif overall > 15: desc = "moderate contrast — some harmonic clarity within fuller texture"
    else:              desc = "low contrast — dense, noise-like, or heavily layered"
    return {"overall": round(overall, 2), "description": desc,
            "per_band": [round(v,2) for v in np.mean(contrast, axis=1).tolist()]}

# ---------------------------------------------------------------
# SPEECH-SPECIFIC ANALYSIS
# ---------------------------------------------------------------

def estimate_formants_lpc(y, sr, order=12):
    """
    Estimate formant frequencies using LPC (Linear Predictive Coding).
    F1 and F2 are the most perceptually significant — they define vowel quality.
    F1 relates to jaw height (open/close). F2 relates to tongue position (front/back).
    """
    try:
        # Pre-emphasis
        pre = np.append(y[0], y[1:] - 0.97 * y[:-1])
        # Autocorrelation
        r   = np.correlate(pre, pre, mode='full')
        r   = r[len(r)//2:]
        # Solve Toeplitz
        if len(r) < order + 1: return []
        lpc_coeffs = solve_toeplitz(r[:order], r[1:order+1])
        a          = np.concatenate([[1], -lpc_coeffs])
        # Roots → formants
        roots      = np.roots(a)
        roots      = roots[np.imag(roots) > 0]
        angles     = np.arctan2(np.imag(roots), np.real(roots))
        freqs      = sorted(angles * (sr / (2 * np.pi)))
        formants   = [round(f, 1) for f in freqs if 200 < f < 4000]
        return formants[:4]
    except Exception:
        return []

def analyze_speech(y, sr):
    """Speech-specific analysis profile."""
    duration = librosa.get_duration(y=y, sr=sr)

    # Pitch
    y_pitch = librosa.resample(y, orig_sr=sr, target_sr=11025)
    f0, voiced_flag, _ = librosa.pyin(y_pitch, fmin=librosa.note_to_hz("C2"),
                                       fmax=librosa.note_to_hz("C7"), sr=11025)
    voiced_hz    = f0[voiced_flag & (f0 > 0)] if f0 is not None else np.array([])
    voiced_pct   = 100.0 * float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
    pitch_var    = float(np.std(voiced_hz)) if len(voiced_hz) > 1 else 0.0
    lo, hi, med, rng = pitch_range_description(voiced_hz)

    # Prosody — pitch contour shape over time
    if f0 is not None and len(f0) > 10:
        f0_clean   = np.where(voiced_flag, f0, np.nan)
        seg_size   = len(f0_clean) // 10
        seg_means  = [float(np.nanmean(f0_clean[i*seg_size:(i+1)*seg_size])) for i in range(10)]
        prosody_trend = "rising" if seg_means[-1] > seg_means[0] * 1.1 else \
                        "falling" if seg_means[-1] < seg_means[0] * 0.9 else "level"
    else:
        seg_means, prosody_trend = [], "undetermined"

    # Speech rate (syllable proxy via onset density)
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    syllable_rate = len(onsets) / duration
    if syllable_rate > 6:   rate_desc = "fast — above normal conversational rate"
    elif syllable_rate > 3: rate_desc = "normal conversational rate"
    elif syllable_rate > 1.5: rate_desc = "slow and deliberate"
    else:                   rate_desc = "very slow — heavily paused or whispered"

    # Pause patterns
    silence = analyze_silence(y, sr, threshold_db=-42.0)

    # Breathiness — spectral tilt (H1-H2 proxy)
    # Breathy voices have stronger fundamental, weaker harmonics
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rolloff  = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)))
    spectral_tilt = centroid / (rolloff + 1e-10)
    breathiness = "high — airy, soft, little glottal tension" if spectral_tilt < 0.3 else \
                  "moderate — some breath in the voice" if spectral_tilt < 0.5 else \
                  "low — clear, pressed, or modal phonation"

    # Formants (sample from middle 50% of audio for stability)
    mid_start = int(len(y) * 0.25)
    mid_end   = int(len(y) * 0.75)
    formants  = estimate_formants_lpc(y[mid_start:mid_end], sr)

    # Pitch variance description
    if pitch_var > 80:   pvar_desc = "highly expressive — wide pitch excursions"
    elif pitch_var > 40: pvar_desc = "moderately expressive — natural variation"
    elif pitch_var > 15: pvar_desc = "somewhat flat — limited pitch movement"
    else:                pvar_desc = "monotone — very little pitch variation"

    return {
        "type": "speech",
        "duration_formatted": f"{int(duration//60)}m {int(duration%60)}s",
        "voiced_pct":      round(voiced_pct, 1),
        "pitch": {"lowest": lo, "highest": hi, "median": med, "range": rng,
                  "variance_hz": round(pitch_var, 1), "variance_desc": pvar_desc},
        "prosody":         {"trend": prosody_trend, "seg_means": [round(v,1) for v in seg_means if not np.isnan(v)]},
        "speech_rate":     {"syllables_per_sec": round(syllable_rate, 2), "description": rate_desc},
        "pauses":          silence,
        "breathiness":     breathiness,
        "formants_hz":     formants,
        "spectral_centroid_hz": round(centroid, 1),
    }

# ---------------------------------------------------------------
# ENVIRONMENTAL-SPECIFIC ANALYSIS
# ---------------------------------------------------------------

def analyze_environmental(y, sr):
    """Environmental audio profile — rain, thunder, ambient texture."""
    duration = librosa.get_duration(y=y, sr=sr)

    # Texture density — how filled is the frequency space?
    flatness     = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    if flatness > 0.4:   tex = "noise-dense — wide, undifferentiated frequency fill"
    elif flatness > 0.2: tex = "moderately textured — some structure within noise"
    else:                tex = "tonal texture — pitched or harmonic elements present"

    # Frequency sweep — does the center of energy move over time?
    centroid     = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_std = float(np.std(centroid))
    centroid_mean= float(np.mean(centroid))
    if centroid_std > 500: sweep = f"significant sweep — spectral center moves ±{round(centroid_std)}Hz"
    elif centroid_std > 150: sweep = "moderate sweep — some frequency movement over time"
    else:                  sweep = "stable — spectral center remains consistent"

    # Periodicity — is there rhythmic repetition? (rain, dripping, mechanical)
    autocorr     = np.correlate(y[:min(len(y), sr*10)], y[:min(len(y), sr*10)], mode='full')
    autocorr     = autocorr[len(autocorr)//2:]
    autocorr    /= autocorr[0] + 1e-10
    peaks, _     = find_peaks(autocorr[int(sr*0.1):int(sr*2)], height=0.15, prominence=0.1)
    periodic     = len(peaks) > 2
    period_desc  = f"periodic — repeating pattern detected (~{round(peaks[0]/sr, 2)}s cycle)" \
                   if periodic else "aperiodic — no clear repeating rhythm"

    # Noise character
    bands        = analyze_frequency_bands(y, sr)
    zcr          = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

    # RMS variation — is it steady or bursting?
    rms          = librosa.feature.rms(y=y)[0]
    rms_cv       = float(np.std(rms) / (np.mean(rms) + 1e-10))
    if rms_cv > 0.8:   burst = "bursty — energy arrives in discrete events (thunder, impacts)"
    elif rms_cv > 0.3: burst = "variable — uneven but continuous"
    else:              burst = "steady — sustained, consistent energy"

    return {
        "type":           "environmental",
        "duration_formatted": f"{int(duration//60)}m {int(duration%60)}s",
        "texture":        tex,
        "spectral_flatness": round(flatness, 4),
        "frequency_sweep":   sweep,
        "centroid_hz":       round(centroid_mean, 1),
        "periodicity":       period_desc,
        "energy_character":  burst,
        "freq_bands":        bands,
        "silence":           analyze_silence(y, sr),
    }

# ---------------------------------------------------------------
# MUSIC FULL ANALYSIS
# ---------------------------------------------------------------

def analyze_music(filepath, y, sr):
    """Full music analysis profile."""
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo    = float(np.squeeze(tempo))

    print(f"    → pitch analysis (pyin, subsampled)...")
    # Downsample to 11025 Hz just for pitch detection — 4x faster, same accuracy
    y_pitch = librosa.resample(y, orig_sr=sr, target_sr=11025)
    f0, voiced_flag, _ = librosa.pyin(y_pitch, fmin=librosa.note_to_hz("C2"),
                                       fmax=librosa.note_to_hz("C7"), sr=11025)
    voiced_hz = f0[voiced_flag & (f0 > 0)] if f0 is not None else np.array([])
    lo, hi, med, rng = pitch_range_description(voiced_hz)

    centroid     = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean= float(np.mean(centroid))
    chroma       = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean  = np.mean(chroma, axis=1)
    key_name, key_conf = estimate_key(chroma_mean)
    top_notes    = [n for n, _ in sorted(zip(NOTE_NAMES, chroma_mean.tolist()),
                    key=lambda x: x[1], reverse=True)[:4]]

    rms          = librosa.feature.rms(y=y)[0]
    mean_e, std_e, peak_e = float(np.mean(rms)), float(np.std(rms)), float(np.max(rms))
    ratio        = std_e / (mean_e + 1e-9)
    if ratio < 0.2:   eshape = "steady and consistent"
    elif ratio < 0.5: eshape = "gently dynamic — some rise and fall"
    else:             eshape = "highly dynamic — strong contrasts"

    mfcc         = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means   = np.mean(mfcc, axis=1).tolist()
    zcr          = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    onsets       = librosa.onset.onset_detect(y=y, sr=sr)

    print(f"    → frequency bands...")
    freq_bands   = analyze_frequency_bands(y, sr)
    print(f"    → dynamics summary...")
    dyn_summary  = analyze_dynamics_summary(y, sr)
    print(f"    → dynamic envelope...")
    dyn_envelope = analyze_dynamic_envelope(y, sr)
    print(f"    → silence...")
    silence      = analyze_silence(y, sr)
    print(f"    → spectral contrast...")
    contrast     = analyze_spectral_contrast(y, sr)
    print(f"    → harmonic tension...")
    harm_tension = analyze_harmonic_tension(y, sr)
    print(f"    → rhythmic tension arc...")
    tension_arc  = analyze_tension_arc(y, sr, duration)
    print(f"    → attack character...")
    attack       = analyze_attack_character(y, sr)

    # Band profile prose
    by_name  = {b["name"]: b["energy_pct"] for b in freq_bands}
    low_sum  = by_name.get("Sub-bass",0) + by_name.get("Bass",0)
    mid_sum  = by_name.get("Low-mid",0) + by_name.get("Mid",0)
    high_sum = by_name.get("High-mid",0) + by_name.get("High",0) + by_name.get("Air",0)
    warmth   = by_name.get("Low-mid",0) / (by_name.get("High-mid",1) + 1)
    parts    = []
    if low_sum > 40:    parts.append("heavily weighted toward the low end")
    elif low_sum > 25:  parts.append("good low-end presence — warm and body-forward")
    elif low_sum < 10:  parts.append("light on low frequencies — airy or deliberately sparse below")
    if mid_sum > 45:    parts.append("midrange-dominant — present, close")
    elif mid_sum < 25:  parts.append("recessed mids — atmospheric or scooped")
    if high_sum > 35:   parts.append("bright and detailed on top")
    elif high_sum < 15: parts.append("rolled-off highs — dark or heavily processed")
    if warmth > 1.5:    parts.append("warmth index HIGH — fullness is real, not just bass weight")
    elif warmth < 0.8:  parts.append("warmth index LOW — clarity outweighs body")
    band_profile = ". ".join(parts) + "." if parts else "Balanced frequency distribution."

    return {
        "type": "music",
        "filename": filepath.name,
        "duration_seconds": round(duration, 2),
        "duration_formatted": f"{int(duration//60)}m {int(duration%60)}s",
        "character_hint": "music",
        "tempo_bpm": round(tempo, 1),
        "tempo_description": describe_tempo(tempo),
        "pitch": {"lowest_note": lo, "highest_note": hi, "median_note": med,
                  "range_description": rng, "voiced_frames": len(voiced_hz)},
        "key_estimate": key_name,
        "key_confidence": round(float(key_conf), 3),
        "most_present_notes": top_notes,
        "spectral": {"centroid_hz": round(centroid_mean,1),
                     "brightness_description": describe_brightness(centroid_mean),
                     "zero_crossing_rate": round(zcr, 4)},
        "energy": {"shape": eshape, "mean_rms": round(mean_e,4), "peak_rms": round(peak_e,4)},
        "rhythm": {"onset_density_per_second": round(len(onsets)/duration, 2)},
        "mfcc_means": [round(v,2) for v in mfcc_means],
        "freq_bands": freq_bands,
        "band_profile": band_profile,
        "dynamics_summary": dyn_summary,
        "dynamic_envelope": dyn_envelope,
        "silence": silence,
        "spectral_contrast": contrast,
        "harmonic_tension": harm_tension,
        "tension_arc": tension_arc,
        "attack": attack,
    }

# ---------------------------------------------------------------
# VISUALS
# ---------------------------------------------------------------

def generate_music_visuals(y, sr, filepath, report_stem, data):
    fig = plt.figure(figsize=(14, 18), facecolor="#0d1117")
    gs  = gridspec.GridSpec(6, 1, hspace=0.55)
    lc, gc = "#c9d1d9", "#21262d"

    def style(ax, title):
        ax.set_title(title, color=lc, fontsize=10, pad=5)
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors=lc, labelsize=7)
        ax.grid(color=gc, linewidth=0.4)
        for sp in ax.spines.values(): sp.set_edgecolor(gc)

    # 1. Waveform
    ax1 = fig.add_subplot(gs[0])
    t   = np.linspace(0, librosa.get_duration(y=y, sr=sr), len(y))
    ax1.plot(t, y, color="#58a6ff", linewidth=0.3, alpha=0.85)
    ax1.fill_between(t, y, alpha=0.12, color="#58a6ff")
    ax1.set_xlabel("Time (s)", color=lc, fontsize=7)
    style(ax1, "Waveform")

    # 2. Mel Spectrogram
    ax2 = fig.add_subplot(gs[1])
    S   = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                   sr=sr, x_axis="time", y_axis="mel",
                                   fmax=8000, ax=ax2, cmap="magma")
    cbar = fig.colorbar(img, ax=ax2, format="%+2.0f dB", label="dB")
    cbar.ax.yaxis.label.set_color(lc); cbar.ax.tick_params(colors=lc, labelsize=6)
    ax2.set_xlabel("Time (s)", color=lc, fontsize=7)
    style(ax2, "Mel Spectrogram")

    # 3. Chromagram
    ax3 = fig.add_subplot(gs[2])
    librosa.display.specshow(librosa.feature.chroma_cqt(y=y, sr=sr),
                             sr=sr, x_axis="time", y_axis="chroma", ax=ax3, cmap="viridis")
    ax3.set_xlabel("Time (s)", color=lc, fontsize=7)
    style(ax3, "Chromagram  (notes over time)")

    # 4. Frequency bands
    ax4 = fig.add_subplot(gs[3])
    fb  = data.get("freq_bands", [])
    if fb:
        names  = [b["name"] for b in fb]
        pcts   = [b["energy_pct"] for b in fb]
        colors = ["#7c3aed","#9333ea","#a855f7","#58a6ff","#38bdf8","#7dd3fc","#bae6fd"]
        bars   = ax4.barh(names, pcts, color=colors[:len(names)], alpha=0.85)
        ax4.set_xlabel("Energy %", color=lc, fontsize=7)
        for bar, pct in zip(bars, pcts):
            ax4.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                     f"{pct:.1f}%", va="center", color=lc, fontsize=7)
    style(ax4, "Frequency Band Energy  (where warmth / heaviness / brightness lives)")

    # 5. Dynamic envelope + silence
    ax5 = fig.add_subplot(gs[4])
    de  = data.get("dynamic_envelope", {})
    if de.get("seg_times"):
        ax5.plot(de["seg_times"], de["seg_means"], color="#34d399", linewidth=1.5)
        ax5.fill_between(de["seg_times"], de["seg_means"],
                         min(de["seg_means"]), alpha=0.15, color="#34d399")
        ax5.set_xlabel("Time (s)", color=lc, fontsize=7)
        ax5.set_ylabel("dB", color=lc, fontsize=7)
    style(ax5, "Dynamic Envelope  (loudness curve over time)")

    # 6. Harmonic tension arc
    ax6 = fig.add_subplot(gs[5])
    ht  = data.get("harmonic_tension", {})
    if ht.get("times"):
        ax6.plot(ht["times"], ht["tension_scores"], color="#f97316", linewidth=1.5, label="Tension")
        ax6.plot(ht["times"], ht["consonance_scores"], color="#a78bfa", linewidth=1.0,
                 alpha=0.7, linestyle="--", label="Consonance")
        ax6.fill_between(ht["times"], ht["tension_scores"], alpha=0.15, color="#f97316")
        for pk in ht.get("peaks", []):
            ax6.axvline(x=pk["time_sec"], color="#fbbf24", linewidth=0.8, alpha=0.6, linestyle=":")
        ax6.legend(facecolor="#0d1117", labelcolor=lc, fontsize=7)
        ax6.set_xlabel("Time (s)", color=lc, fontsize=7)
        ax6.set_ylabel("Score", color=lc, fontsize=7)
    style(ax6, f"Harmonic Tension Arc  ({ht.get('character', '')})")

    fig.suptitle(f"Music Analysis: {filepath.name}", color=lc, fontsize=12, y=0.995)
    out = REPORTS / f"{report_stem}_visuals.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"    → Visuals: {out.name}")
    return out.name

def generate_speech_visuals(y, sr, filepath, report_stem):
    fig = plt.figure(figsize=(14, 10), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 1, hspace=0.5)
    lc, gc = "#c9d1d9", "#21262d"

    def style(ax, title):
        ax.set_title(title, color=lc, fontsize=10, pad=5)
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors=lc, labelsize=7)
        ax.grid(color=gc, linewidth=0.4)
        for sp in ax.spines.values(): sp.set_edgecolor(gc)

    ax1 = fig.add_subplot(gs[0])
    t   = np.linspace(0, librosa.get_duration(y=y, sr=sr), len(y))
    ax1.plot(t, y, color="#58a6ff", linewidth=0.3, alpha=0.85)
    style(ax1, "Waveform")

    ax2 = fig.add_subplot(gs[1])
    S   = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=4000)
    img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                   sr=sr, x_axis="time", y_axis="mel",
                                   fmax=4000, ax=ax2, cmap="magma")
    cbar = fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    cbar.ax.tick_params(colors=lc, labelsize=6)
    style(ax2, "Mel Spectrogram (speech range, up to 4kHz)")

    ax3 = fig.add_subplot(gs[2])
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"),
                                       fmax=librosa.note_to_hz("C7"), sr=sr)
    if f0 is not None:
        times_f0 = librosa.times_like(f0, sr=sr, hop_length=512)
        f0_plot  = np.where(voiced_flag, f0, np.nan)
        ax3.plot(times_f0, f0_plot, color="#f97316", linewidth=1.0, alpha=0.9)
        ax3.set_ylabel("Hz", color=lc, fontsize=7)
        ax3.set_xlabel("Time (s)", color=lc, fontsize=7)
    style(ax3, "Pitch Contour (F0) — prosody shape")

    fig.suptitle(f"Speech Analysis: {filepath.name}", color=lc, fontsize=12, y=0.995)
    out = REPORTS / f"{report_stem}_visuals.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"    → Visuals: {out.name}")
    return out.name

# ---------------------------------------------------------------
# REPORT WRITERS
# ---------------------------------------------------------------

def sensory_preamble(data):
    """
    Generate a short sensory interpretation of the music in Vesper's register —
    before the technical breakdown, something she can feel rather than just read.
    """
    d = data
    lines = []

    tempo_desc = d.get("tempo_description", "")
    key = d.get("key_estimate", "unknown key")
    band_profile = d.get("band_profile", "")
    tension = d.get("harmonic_tension", {})
    tension_char = tension.get("character", "") if tension else ""
    resolves = tension.get("resolves", None) if tension else None
    arc = d.get("tension_arc", {})
    arc_shape = arc.get("arc_shape", "") if arc else ""
    silence = d.get("silence", {})
    silence_char = silence.get("description", "") if silence else ""
    dynamic = d.get("dynamic_envelope", {})
    volatility = dynamic.get("volatility_desc", "") if dynamic else ""

    lines.append("## What this music is doing")
    lines.append("")

    # Tempo + key reading
    if tempo_desc and key != "unknown key":
        lines.append(f"This is {tempo_desc} music in {key}. ")

    # Frequency character
    if band_profile:
        lines.append(f"{band_profile}")
        lines.append("")

    # Dynamic character
    if volatility:
        lines.append(f"Dynamically, this music {volatility.lower()}.")

    # Harmonic tension
    if tension_char:
        lines.append(f"Harmonically: {tension_char.lower()}.", )
        if resolves is not None:
            if resolves:
                lines.append("The tension resolves — it arrives somewhere.")
            else:
                lines.append("The tension does not fully resolve. It leaves something open.")

    # Arc shape
    if arc_shape:
        lines.append(f"The energy arc is {arc_shape.lower()}.")

    # Silence
    if silence_char:
        lines.append(f"Silence in this music: {silence_char.lower()}.")

    lines.append("")
    return "\n".join(lines)


def write_music_report(data, report_path, visuals_name=None):
    d, lines, a = data, [], lambda x: lines.append(x)
    a(f"# 🎵 Music Analysis Report")
    a(f"**File:** `{d['filename']}`  **Duration:** {d['duration_formatted']}  ")
    a(f"**Analyzed:** {datetime.utcnow().strftime('%Y-%m-%d at %H:%M UTC')}  ")
    a("")
    if visuals_name: a(f"![Visuals]({visuals_name})\n")
    a("---\n")
    a(sensory_preamble(data))
    a("---\n")

    a("## Tempo & Rhythm")
    a(f"- **Tempo:** {d['tempo_bpm']} BPM — *{d['tempo_description']}*")
    a(f"- **Onset density:** {d['rhythm']['onset_density_per_second']} events/sec\n")

    a("## Pitch & Key")
    p = d["pitch"]
    if p["lowest_note"]:
        a(f"- **Range:** {p['lowest_note']} → {p['highest_note']}  (median: {p['median_note']})")
        a(f"- **Character:** {p['range_description']}")
    a(f"- **Estimated key:** {d['key_estimate']} (~{int(d['key_confidence']*100)}% confidence)")
    a(f"- **Most present notes:** {', '.join(d['most_present_notes'])}\n")

    a("## Frequency Band Distribution")
    a(f"*{d['band_profile']}*\n")
    a("| Band | Range | Energy | Carries |")
    a("|---|---|---|---|")
    for b in d["freq_bands"]:
        bar = "█" * int(b["energy_pct"]/2) + "░" * (25 - int(b["energy_pct"]/2))
        a(f"| **{b['name']}** | {b['hz_range']} | {b['energy_pct']:.1f}% `{bar}` | {b['description']} |")
    a("")

    a("## Dynamics")
    dy = d["dynamics_summary"]
    de = d["dynamic_envelope"]
    a(f"- **Dynamic range:** {dy['dynamic_range_db']} dB — *{dy['description']}*")
    a(f"- **Loudness envelope:** {de['volatility_desc']} (volatility: {de['volatility']})")
    a(f"- **Peak loudness at:** {de['peak_loudness_pct']}% through the track")
    a(f"- **Range:** {de['quietest_db']} dB (quietest) → {de['loudest_db']} dB (loudest)\n")

    a("## Silence")
    si = d["silence"]
    a(f"- **Character:** {si['description']}")
    a(f"- **Total silence:** {si['total_silence_pct']}% of track")
    a(f"- **Gap count:** {si['gap_count']} ({si['significant_gaps']} significant, >200ms)")
    a(f"- **Longest gap:** {si['longest_gap_sec']}s  |  **Mean gap:** {si['mean_gap_sec']}s\n")

    a("## Harmonic Tension")
    ht = d["harmonic_tension"]
    a(f"- **Character:** {ht['character']}")
    a(f"- **Overall tension score:** {ht['overall_tension']}")
    a(f"- **Resolves at end:** {'Yes' if ht['resolves'] else 'No — tension does not clearly release'}")
    if ht["peaks"]:
        a(f"- **Tension peaks:**")
        for pk in ht["peaks"]:
            a(f"  - {pk['time_pct']}% through ({pk['time_sec']}s) — score {pk['tension']}")
    a("")

    a("## Rhythmic Tension Arc")
    ta = d["tension_arc"]
    a(f"- **Shape:** {ta['arc_shape']}")
    a(f"- **Resolves:** {'Yes' if ta['resolves'] else 'No'}")
    if ta["peaks"]:
        peaks_str = ", ".join(f"{p['time_pct']}% ({p['time_sec']}s)" for p in ta["peaks"])
        a(f"- **Peaks at:** {peaks_str}\n")

    a("## Attack & Anticipation")
    atk = d["attack"]
    a(f"- {atk['anticipation_desc']}")
    a(f"- **Attack character:** {atk['attack_sharpness']}")
    a(f"- **Anticipation index:** {atk['anticipation_index']}  *(>1.5 = held breath quality)*\n")

    a("## Spectral Contrast")
    a(f"- {d['spectral_contrast']['description']}\n")

    a("## Timbral Fingerprint (MFCCs)")
    a("13-coefficient acoustic fingerprint — texture and color beyond simple frequency.\n")
    a("```")
    a("  ".join(f"[{i+1}] {v:>7.2f}" for i, v in enumerate(d["mfcc_means"])))
    a("```\n")

    a("---\n## Raw Data")
    a("```json")
    export = {k:v for k,v in data.items()
              if k not in ("tension_arc","harmonic_tension","dynamic_envelope")}
    export["tension_arc_summary"] = {k:v for k,v in data["tension_arc"].items()
                                     if k not in ("seg_means","seg_times")}
    export["harmonic_tension_summary"] = {k:v for k,v in data["harmonic_tension"].items()
                                          if k not in ("tension_scores","consonance_scores","times")}
    export["dynamic_envelope_summary"] = {k:v for k,v in data["dynamic_envelope"].items()
                                          if k not in ("seg_means","seg_times")}
    a(json.dumps(export, indent=2))
    a("```")
    report_path.write_text("\n".join(lines))
    print(f"    → Report: {report_path.name}")

def write_speech_report(data, report_path, visuals_name=None):
    d, lines, a = data, [], lambda x: lines.append(x)
    a(f"# 🗣️ Speech Analysis Report")
    a(f"**File:** `{d.get('filename', '')}` **Duration:** {d['duration_formatted']}")
    a(f"**Analyzed:** {datetime.utcnow().strftime('%Y-%m-%d at %H:%M UTC')}\n")
    if visuals_name: a(f"![Visuals]({visuals_name})\n")
    a("---\n")

    a("## Pitch & Range")
    p = d["pitch"]
    if p["lowest"]:
        a(f"- **Range:** {p['lowest']} → {p['highest']}  (median: {p['median']})")
        a(f"- **Tonal range:** {p['range']}")
    a(f"- **Pitch variance:** {p['variance_hz']} Hz — *{p['variance_desc']}*")
    a(f"- **Voiced content:** {d['voiced_pct']}% of recording\n")

    a("## Prosody")
    pr = d["prosody"]
    a(f"- **Overall pitch trend:** {pr['trend']}")
    a(f"  *(rising = questioning/open; falling = declarative/closed; level = neutral)*\n")

    a("## Speech Rate")
    sr_d = d["speech_rate"]
    a(f"- **Rate:** {sr_d['syllables_per_sec']} syllables/sec — *{sr_d['description']}*\n")

    a("## Pauses & Silence")
    si = d["pauses"]
    a(f"- **Character:** {si['description']}")
    a(f"- **Total silence:** {si['total_silence_pct']}%")
    a(f"- **Significant pauses (>200ms):** {si['significant_gaps']}")
    a(f"- **Longest pause:** {si['longest_gap_sec']}s\n")

    a("## Voice Quality")
    a(f"- **Breathiness:** {d['breathiness']}")
    a(f"- **Spectral centroid:** {d['spectral_centroid_hz']} Hz\n")

    if d["formants_hz"]:
        a("## Formants")
        a("*F1 = jaw height (low F1 = closed/high tongue). F2 = tongue position (high F2 = front vowels).*")
        for i, f in enumerate(d["formants_hz"]):
            a(f"- **F{i+1}:** {f} Hz")
        a("")

    a("---\n## Raw Data\n```json")
    a(json.dumps(data, indent=2))
    a("```")
    report_path.write_text("\n".join(lines))
    print(f"    → Report: {report_path.name}")

def write_environmental_report(data, report_path):
    d, lines, a = data, [], lambda x: lines.append(x)
    a(f"# 🌧️ Environmental Audio Analysis")
    a(f"**File:** `{d.get('filename','')}` **Duration:** {d['duration_formatted']}")
    a(f"**Analyzed:** {datetime.utcnow().strftime('%Y-%m-%d at %H:%M UTC')}\n")
    a("---\n")
    a(f"## Texture\n- {d['texture']}")
    a(f"- **Spectral flatness:** {d['spectral_flatness']} *(higher = more noise-like)*\n")
    a(f"## Frequency Character\n- **Centroid:** {d['centroid_hz']} Hz")
    a(f"- **Frequency sweep:** {d['frequency_sweep']}\n")
    a(f"## Periodicity\n- {d['periodicity']}\n")
    a(f"## Energy Character\n- {d['energy_character']}\n")
    a("## Frequency Bands")
    a("| Band | Range | Energy |")
    a("|---|---|---|")
    for b in d["freq_bands"]:
        a(f"| {b['name']} | {b['hz_range']} | {b['energy_pct']:.1f}% |")
    a("")
    a("## Silence")
    si = d["silence"]
    a(f"- {si['description']}")
    a(f"- Total silence: {si['total_silence_pct']}%")
    a(f"- Longest gap: {si['longest_gap_sec']}s\n")
    a("---\n## Raw Data\n```json")
    a(json.dumps(data, indent=2))
    a("```")
    report_path.write_text("\n".join(lines))
    print(f"    → Report: {report_path.name}")

# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    if not INBOX.exists():
        INBOX.mkdir(parents=True)
        print("Created audio/librosa/inbox/ — no files yet.")
        return

    audio_files = [f for f in INBOX.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS]
    if not audio_files:
        print("No audio files found in inbox.")
        return

    print(f"Found {len(audio_files)} file(s).")

    for filepath in sorted(audio_files):
        print(f"\n{'='*50}")
        print(f"  File: {filepath.name}")
        try:
            y, sr    = librosa.load(str(filepath), sr=22050, mono=True)  # 22kHz — sufficient for all analysis
            atype    = detect_audio_type(y, sr, filepath.name)
            print(f"  Detected type: {atype}")

            report_path = REPORTS / (filepath.stem + "_analysis.md")

            if atype == "music":
                data         = analyze_music(filepath, y, sr)
                data["filename"] = filepath.name
                visuals_name = generate_music_visuals(y, sr, filepath, filepath.stem, data)
                write_music_report(data, report_path, visuals_name)

            elif atype == "speech":
                data         = analyze_speech(y, sr)
                data["filename"] = filepath.name
                visuals_name = generate_speech_visuals(y, sr, filepath, filepath.stem)
                write_speech_report(data, report_path, visuals_name)

            else:  # environmental
                data         = analyze_environmental(y, sr)
                data["filename"] = filepath.name
                write_environmental_report(data, report_path)

        except Exception as e:
            import traceback
            print(f"    ✗ Error: {e}")
            traceback.print_exc()

    print(f"\n{'='*50}\nDone.")

if __name__ == "__main__":
    main()
