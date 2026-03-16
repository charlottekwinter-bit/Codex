import os
import re
import requests

API_KEY = os.environ.get("LASTFM_API_KEY")
BASE_URL = "http://ws.audioscrobbler.com/2.0/"
WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary"


def read_request():
    with open("Audio/artist-request.txt", "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        raise ValueError("artist-request.txt is empty. Write an artist name.")
    return lines[0].strip()


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


def get_top_tracks(artist):
    params = {
        "method": "artist.getTopTracks",
        "api_key": API_KEY,
        "artist": artist,
        "format": "json",
        "autocorrect": 1,
        "limit": 8,
    }
    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    return r.json().get("toptracks", {}).get("track", [])


def get_top_albums(artist):
    params = {
        "method": "artist.getTopAlbums",
        "api_key": API_KEY,
        "artist": artist,
        "format": "json",
        "autocorrect": 1,
        "limit": 5,
    }
    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    return r.json().get("topalbums", {}).get("album", [])


def get_similar_artists(artist):
    params = {
        "method": "artist.getSimilar",
        "api_key": API_KEY,
        "artist": artist,
        "format": "json",
        "autocorrect": 1,
        "limit": 6,
    }
    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    return r.json().get("similarartists", {}).get("artist", [])


def get_wiki_summary(artist):
    try:
        r = requests.get(f"{WIKI_API}/{artist.replace(' ', '_')}", timeout=10)
        if r.status_code == 200:
            data = r.json()
            extract = data.get("extract", "")
            extract = re.sub(r'\s+', ' ', extract).strip()
            if len(extract) > 600:
                extract = extract[:600].rsplit('.', 1)[0] + '.'
            return extract
    except Exception:
        pass
    return None


def clean_bio(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text).strip()
    text = text.split("Read more")[0].strip()
    text = text.split("User-contributed text")[0].strip()
    if len(text) > 600:
        text = text[:600].rsplit('. ', 1)[0] + '.'
    return text


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


def format_response(artist_name, info, tracks, albums, similar):
    lines = []

    # Corrected name from Last.fm
    name = info.get("name", artist_name)
    listeners = info.get("stats", {}).get("listeners", "")
    playcount = info.get("stats", {}).get("playcount", "")

    lines.append(name)
    if listeners:
        lines.append(f"  {format_listeners(listeners)} listeners · {format_listeners(playcount)} plays")
    lines.append("")

    # Tags
    tags_raw = info.get("tags", {}).get("tag", [])
    tags = [t["name"] for t in tags_raw[:8]] if tags_raw else []
    if tags:
        lines.append(f"Tagged as: {', '.join(tags)}.")
        lines.append("")

    # Biography
    bio = clean_bio(info.get("bio", {}).get("summary", ""))
    if not bio:
        bio = get_wiki_summary(name)
    if bio:
        lines.append(bio)
        lines.append("")

    # Similar artists
    if similar:
        sim_names = [a["name"] for a in similar[:5]]
        lines.append(f"Similar artists: {', '.join(sim_names)}.")
        lines.append("")

    # Top tracks
    if tracks:
        track_names = [t["name"] for t in tracks[:6]]
        lines.append(f"Most listened tracks: {', '.join(track_names)}.")

    # Top albums
    if albums:
        # Filter out "[unknown album]" entries
        album_names = [a["name"] for a in albums[:5] if a.get("name") and "[" not in a["name"]]
        if album_names:
            lines.append(f"Notable albums: {', '.join(album_names)}.")

    return "\n".join(lines)


def main():
    artist = read_request()
    print(f"Looking up artist: {artist}")

    info = get_artist_info(artist)
    if not info:
        response = f"Artist not found: {artist}\n\nCheck the spelling and try again."
    else:
        tracks = get_top_tracks(artist)
        albums = get_top_albums(artist)
        similar = get_similar_artists(artist)
        response = format_response(artist, info, tracks, albums, similar)

    with open("Audio/artist-response.txt", "w") as f:
        f.write(response)

    print("Response written to Audio/artist-response.txt")
    print("---")
    print(response)


if __name__ == "__main__":
    main()
