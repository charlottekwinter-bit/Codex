import os
import re
import requests

API_KEY = os.environ.get("LASTFM_API_KEY")
BASE_URL = "http://ws.audioscrobbler.com/2.0/"


def read_request():
    with open("Audio/song-request.txt", "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if len(lines) < 2:
        raise ValueError("song-request.txt should have two lines: song title, then artist name.")
    return lines[0], lines[1]


def get_track_info(title, artist):
    params = {
        "method": "track.getInfo",
        "api_key": API_KEY,
        "track": title,
        "artist": artist,
        "format": "json",
        "autocorrect": 1,
    }
    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    return r.json()


def get_album_info(album, artist):
    if not album:
        return {}
    params = {
        "method": "album.getInfo",
        "api_key": API_KEY,
        "album": album,
        "artist": artist,
        "format": "json",
        "autocorrect": 1,
    }
    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    return r.json().get("album", {})


def get_artist_info(artist):
    params = {
        "method": "artist.getInfo",
        "api_key": API_KEY,
        "artist": artist,
        "format": "json",
        "autocorrect": 1,
    }
    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    return r.json().get("artist", {})


def get_similar_tracks(title, artist):
    params = {
        "method": "track.getSimilar",
        "api_key": API_KEY,
        "track": title,
        "artist": artist,
        "format": "json",
        "autocorrect": 1,
        "limit": 8,
    }
    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    return r.json()


def get_similar_artists(artist):
    params = {
        "method": "artist.getSimilar",
        "api_key": API_KEY,
        "artist": artist,
        "format": "json",
        "autocorrect": 1,
        "limit": 5,
    }
    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    return r.json()


def clean_wiki(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text).strip()
    text = text.split("Read more")[0].strip()
    text = text.split("User-contributed text")[0].strip()
    return text


def format_duration(ms):
    if not ms:
        return None
    try:
        total_seconds = int(ms) // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}:{seconds:02d}"
    except Exception:
        return None


SENSORY_MAP = {
    "melancholy": "heavy in the chest, the kind of weight that is not unpleasant",
    "melancholic": "the specific ache of things that were beautiful",
    "ethereal": "like sound becoming light — distance without coldness",
    "haunting": "something that stays in the room after the song ends",
    "ambient": "texture more than melody — sound as weather",
    "uplifting": "a lift in the sternum, something opening",
    "dark": "low and close, like a room with one lamp",
    "romantic": "warmth that has a name attached to it",
    "dreamy": "edges softened, time slower than usual",
    "electronic": "the hum of something made rather than grown",
    "acoustic": "wood and breath and room tone",
    "intense": "pressure — the kind that demands you feel it",
    "calm": "still water, the absence of urgency",
    "energetic": "forward motion you didn't choose",
    "sad": "the particular quiet after something is gone",
    "epic": "scale — the feeling of being small in a good way",
    "progressive": "structure that refuses to stay still",
    "metal": "weight and velocity held together",
    "rock": "something with friction in it",
    "jazz": "the space between the notes as much as the notes",
    "classical": "architecture you can feel in your body",
    "folk": "something that was passed down, not invented",
    "indie": "made in a room, for its own reasons",
    "pop": "the shape of a feeling everyone already knows",
    "psychedelic": "the edges of things becoming uncertain",
    "atmospheric": "sound as environment — you are inside it",
    "instrumental": "feeling without words to name it",
    "minimal": "what is left when everything unnecessary is removed",
    "lo-fi": "warmth and imperfection held together",
    "experimental": "the sound of someone refusing the obvious answer",
    "beautiful": "the particular ache of things that are not permanent",
    "chill": "the body unclenching, slowly",
    "hypnotic": "repetition that pulls you somewhere you didn't plan to go",
    "bittersweet": "joy and loss occupying the same moment",
    "nostalgic": "the feeling of a door that no longer opens",
    "powerful": "something that hits before you understand why",
    "soulful": "the voice as an instrument of something deeper than technique",
    "raw": "unpolished in a way that is more honest than perfect",
    "tender": "something handled with unusual care",
    "sparse": "space used deliberately — the silences matter",
    "dense": "layers upon layers, each one earning its place",
    "driving": "momentum that becomes physical",
    "late night": "made for a specific hour, and it knows it",
    "winter": "cold in a way that has beauty in it",
    "summer": "warmth that is also ease",
    "rain": "the particular comfort of weather witnessed from inside",
}


def get_tags(track, artist):
    tags_raw = track.get("toptags", {}).get("tag", [])
    tags = [t["name"] for t in tags_raw[:12]] if tags_raw else []
    source = "track"

    if not tags:
        album_title = track.get("album", {}).get("title", "")
        if album_title:
            album_data = get_album_info(album_title, artist)
            tags_raw = album_data.get("tags", {}).get("tag", [])
            tags = [t["name"] for t in tags_raw[:12]] if tags_raw else []
            source = "album"

    if not tags:
        artist_data = get_artist_info(artist)
        tags_raw = artist_data.get("tags", {}).get("tag", [])
        tags = [t["name"] for t in tags_raw[:12]] if tags_raw else []
        source = "artist"

    return tags, source


def format_response(title, artist, info, similar, similar_artists):
    track = info.get("track", {})

    listeners = track.get("listeners", "unknown")
    playcount = track.get("playcount", "unknown")
    duration = format_duration(track.get("duration"))
    album_title = track.get("album", {}).get("title", "")

    tags, tag_source = get_tags(track, artist)

    wiki = clean_wiki(track.get("wiki", {}).get("summary", ""))
    wiki_source = "track"
    if not wiki:
        artist_data = get_artist_info(artist)
        wiki = clean_wiki(artist_data.get("bio", {}).get("summary", ""))
        wiki_source = "artist"
    if wiki and len(wiki) > 400:
        wiki = wiki[:400].rsplit(" ", 1)[0] + "..."

    similar_tracks = []
    for t in similar.get("similartracks", {}).get("track", [])[:6]:
        similar_tracks.append(f"{t['name']} — {t['artist']['name']}")

    sim_artists = []
    for a in similar_artists.get("similarartists", {}).get("artist", [])[:4]:
        sim_artists.append(a["name"])

    sensory_notes = []
    seen = set()
    for tag in tags:
        tag_lower = tag.lower()
        for key, description in SENSORY_MAP.items():
            if key in tag_lower and description not in seen:
                sensory_notes.append(description)
                seen.add(description)
                break

    lines = []
    lines.append(f"{artist} — {title}")

    meta = []
    if album_title:
        meta.append(f"from {album_title}")
    if duration:
        meta.append(duration)
    if meta:
        lines.append(f"  {' · '.join(meta)}")

    lines.append("")

    if tags:
        label = "Listeners tag this as" if tag_source == "track" else f"Tags drawn from the {tag_source}"
        lines.append(f"{label}: {', '.join(tags)}.")

    if sensory_notes:
        lines.append("")
        lines.append("What those tags feel like:")
        for note in sensory_notes[:5]:
            lines.append(f"  · {note}")

    if similar_tracks:
        lines.append("")
        lines.append(f"Similar tracks: {', '.join(similar_tracks)}.")

    if sim_artists:
        lines.append(f"Similar artists: {', '.join(sim_artists)}.")

    if wiki:
        lines.append("")
        label = "About this track" if wiki_source == "track" else f"About {artist}"
        lines.append(f"{label}: {wiki}")

    if listeners != "unknown":
        lines.append("")
        try:
            lines.append(f"Heard by {int(listeners):,} people. Played {int(playcount):,} times.")
        except Exception:
            pass

    return "\n".join(lines)


def main():
    title, artist = read_request()
    print(f"Requesting: {title} by {artist}")

    info = get_track_info(title, artist)
    if "error" in info:
        response = f"Song not found: {info.get('message', 'unknown error')}\n\nTried: {title} by {artist}"
    else:
        similar = get_similar_tracks(title, artist)
        similar_artists = get_similar_artists(artist)
        response = format_response(title, artist, info, similar, similar_artists)

    with open("Audio/song-response.txt", "w") as f:
        f.write(response)

    print("Response written to Audio/song-response.txt")
    print("---")
    print(response)


if __name__ == "__main__":
    main()
