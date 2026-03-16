import os
import re
import requests

API_KEY = os.environ.get("LASTFM_API_KEY")
BASE_URL = "http://ws.audioscrobbler.com/2.0/"

SENSORY_MAP = {
    "trip-hop": "heavy and slow-moving, like weather inside a room",
    "ambient": "texture without edges — sound as atmosphere",
    "electronic": "constructed rather than played, precise and cool",
    "post-rock": "builds and releases, instrumental and patient",
    "shoegaze": "guitars as walls, vocals as something underneath",
    "dream pop": "soft focus, emotionally blurred at the edges",
    "darkwave": "cold light, minor keys, something beautiful and sad",
    "gothic rock": "theatrical darkness, dramatic and ornate",
    "industrial": "mechanical and aggressive, rhythm as percussion against structure",
    "noise": "texture pushed past comfort into something else",
    "classical": "formal structure as a container for feeling",
    "jazz": "conversation between instruments, structure as negotiation",
    "blues": "feeling before technique, rawness as the point",
    "soul": "voice as the primary instrument, emotion made physical",
    "r&b": "rhythm and warmth, groove as architecture",
    "hip-hop": "language as rhythm, the spoken word in motion",
    "rap": "words at speed, density of meaning per line",
    "folk": "acoustic and intimate, the voice in a room",
    "country": "storytelling with specific geography and time",
    "pop": "immediate and accessible, hooks as the structure",
    "rock": "electric and forward-moving, weight and momentum",
    "punk": "fast and short and deliberate, rejection as aesthetic",
    "metal": "loud and precise, heaviness as emotional vocabulary",
    "post-punk": "angular and tense, rhythm-forward, ideas over comfort",
    "new wave": "synthesizers as texture, detached and danceable",
    "psychedelic": "altered perception as structure, the mind as landscape",
    "experimental": "rules suspended, form as question rather than answer",
    "minimalist": "repetition as transformation, small changes as event",
    "baroque": "elaborate and formal, ornamentation as meaning",
    "orchestral": "scale and layering, many voices as one",
    "singer-songwriter": "intimate and confessional, the person behind the voice",
    "acoustic": "wood and string and air, nothing between you and the sound",
    "lo-fi": "texture and imperfection as warmth, fidelity as choice",
    "indie": "made on its own terms, outside the commercial frame",
    "alternative": "adjacent to mainstream but not inside it",
    "britpop": "guitar pop with specific English geography",
    "grunge": "heavy and raw, distortion as emotional state",
    "emo": "earnest and emotionally exposed, volume as feeling",
    "post-metal": "slow and heavy and searching, metal as meditation",
    "drone": "sustained tone as hypnosis, time dissolved",
    "spoken word": "language without music's structure, the voice alone",
    "world": "geography as sound, specific place encoded in rhythm",
    "latin": "syncopation and warmth, the body in the beat",
    "reggae": "the offbeat as the point, space as ingredient",
    "dub": "echo and absence, the remix as original",
    "techno": "machine rhythm, the body as instrument in a system",
    "house": "the four-four pulse as floor, movement as intent",
    "trance": "repetition as ascent, the long build",
    "downtempo": "slow bpm, cool and interior, music for thinking inside",
}


def read_request():
    with open("Audio/album-request.txt", "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        raise ValueError("album-request.txt is empty.")
    if len(lines) >= 2:
        return lines[0], lines[1]
    parts = lines[0].rsplit(" / ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return lines[0], None


def get_album_info(album, artist):
    params = {
        "method": "album.getInfo",
        "api_key": API_KEY,
        "album": album,
        "format": "json",
        "autocorrect": 1,
    }
    if artist:
        params["artist"] = artist
    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    return r.json().get("album", {})


def get_artist_tags(artist):
    params = {
        "method": "artist.getTopTags",
        "api_key": API_KEY,
        "artist": artist,
        "format": "json",
        "autocorrect": 1,
    }
    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    return r.json().get("toptags", {}).get("tag", [])


def clean_wiki(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text).strip()
    text = text.split("Read more")[0].strip()
    text = text.split("User-contributed text")[0].strip()
    if len(text) > 500:
        text = text[:500].rsplit('. ', 1)[0] + '.'
    return text


def format_duration(seconds):
    try:
        s = int(seconds)
        m, sec = divmod(s, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m}m"
        return f"{m}m {sec:02d}s"
    except Exception:
        return ""


def format_listeners(n):
    try:
        n = int(n)
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        elif n >= 1_000:
            return f"{n/1_000:.0f}K"
        return str(n)
    except Exception:
        return str(n)


def get_sensory_notes(tags):
    notes = []
    seen = set()
    for tag in tags:
        name = tag.get("name", "").lower()
        for key, desc in SENSORY_MAP.items():
            if key in name and key not in seen:
                notes.append(f"{key}: {desc}")
                seen.add(key)
            if len(notes) >= 4:
                break
        if len(notes) >= 4:
            break
    return notes


def format_response(info, extra_tags=None):
    name = info.get("name", "")
    artist = info.get("artist", "")
    listeners = info.get("listeners", "")
    playcount = info.get("playcount", "")

    tags_raw = info.get("tags", {}).get("tag", [])
    if not tags_raw and extra_tags:
        tags_raw = extra_tags
    tags = [t["name"] for t in tags_raw[:10]] if tags_raw else []

    tracks_raw = info.get("tracks", {}).get("track", [])
    wiki_text = clean_wiki(info.get("wiki", {}).get("summary", ""))

    lines = []
    lines.append(f"{artist} — {name}")
    lines.append("")

    if listeners:
        lines.append(f"  {format_listeners(listeners)} listeners · {format_listeners(playcount)} plays")
        lines.append("")

    if tags:
        lines.append(f"Tagged as: {', '.join(tags)}.")
        lines.append("")

    sensory = get_sensory_notes(tags_raw if tags_raw else (extra_tags or []))
    if sensory:
        lines.append("What this sounds like:")
        for note in sensory:
            lines.append(f"  {note}")
        lines.append("")

    if wiki_text:
        lines.append(wiki_text)
        lines.append("")

    if tracks_raw:
        lines.append("—" * 40)
        lines.append("")
        lines.append("Tracks:")
        lines.append("")
        total_duration = 0
        for i, track in enumerate(tracks_raw, 1):
            tname = track.get("name", "")
            dur = track.get("duration", "")
            dur_str = format_duration(dur) if dur and dur != "0" else ""
            try:
                total_duration += int(dur)
            except Exception:
                pass
            if dur_str:
                lines.append(f"  {i:2}. {tname}  [{dur_str}]")
            else:
                lines.append(f"  {i:2}. {tname}")

        if total_duration > 0:
            lines.append("")
            lines.append(f"  Total: {format_duration(total_duration)}")

    return "\n".join(lines)


def main():
    album, artist = read_request()
    print(f"Looking up album: {album}" + (f" by {artist}" if artist else ""))

    info = get_album_info(album, artist)
    if not info:
        response = f"Album not found: {album}\n\nCheck the spelling and try again."
    else:
        extra_tags = []
        actual_artist = info.get("artist", artist)
        if actual_artist and not info.get("tags", {}).get("tag"):
            try:
                extra_tags = get_artist_tags(actual_artist)
            except Exception:
                pass
        response = format_response(info, extra_tags)

    with open("Audio/album-response.txt", "w") as f:
        f.write(response)

    print("Response written to Audio/album-response.txt")
    print("---")
    print(response[:400])


if __name__ == "__main__":
    main()
