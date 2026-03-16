import os
import re
import requests

LYRICS_OVH = "https://api.lyrics.ovh/v1"
LASTFM_URL = "http://ws.audioscrobbler.com/2.0/"
API_KEY = os.environ.get("LASTFM_API_KEY")


def read_request():
    with open("Audio/lyrics-request.txt", "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if len(lines) < 2:
        raise ValueError("lyrics-request.txt should have two lines: song title, then artist name.")
    return lines[0], lines[1]


def get_lyrics_ovh(title, artist):
    """Primary source: lyrics.ovh"""
    try:
        r = requests.get(f"{LYRICS_OVH}/{artist}/{title}", timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        lyrics = data.get("lyrics", None)
        return lyrics if lyrics and len(lyrics.strip()) > 20 else None
    except Exception as e:
        print(f"lyrics.ovh failed: {e}")
        return None


def get_lyrics_chartlyrics(title, artist):
    """Fallback: chartlyrics (SOAP-based but has a simple search endpoint)"""
    try:
        r = requests.get(
            "http://api.chartlyrics.com/apiv1.asmx/SearchLyricDirect",
            params={"artist": artist, "song": title},
            timeout=10
        )
        if r.ok and "<Lyric>" in r.text:
            match = re.search(r"<Lyric>(.*?)</Lyric>", r.text, re.DOTALL)
            if match:
                lyrics = match.group(1).strip()
                return lyrics if len(lyrics) > 20 else None
    except Exception as e:
        print(f"chartlyrics failed: {e}")
    return None


def get_song_context(title, artist):
    """Pull track metadata from Last.fm to give lyrics context."""
    if not API_KEY:
        return {}
    try:
        r = requests.get(LASTFM_URL, params={
            "method": "track.getInfo",
            "api_key": API_KEY,
            "track": title,
            "artist": artist,
            "format": "json",
            "autocorrect": 1,
        }, timeout=10)
        if r.ok:
            data = r.json()
            if "error" not in data:
                return data.get("track", {})
    except Exception as e:
        print(f"Last.fm context fetch failed: {e}")
    return {}


def clean_lyrics(raw):
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned = []
    blank_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped == "":
            blank_count += 1
            if blank_count <= 1:
                cleaned.append("")
        else:
            blank_count = 0
            cleaned.append(stripped)
    while cleaned and cleaned[0] == "":
        cleaned.pop(0)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    return "\n".join(cleaned)


def clean_wiki(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text).strip()
    text = text.split("Read more")[0].strip()
    text = text.split("User-contributed text")[0].strip()
    if len(text) > 300:
        text = text[:300].rsplit(" ", 1)[0] + "..."
    return text


def format_response(title, artist, lyrics, context):
    lines = []
    lines.append(f"{artist} — {title}")

    # Album and tags from context
    album_title = context.get("album", {}).get("title", "")
    if album_title:
        lines.append(f"  from {album_title}")

    tags_raw = context.get("toptags", {}).get("tag", [])
    tags = [t["name"] for t in tags_raw[:6]] if tags_raw else []
    if tags:
        lines.append(f"  tagged: {', '.join(tags)}")

    wiki = clean_wiki(context.get("wiki", {}).get("summary", ""))
    if wiki:
        lines.append("")
        lines.append(wiki)

    lines.append("")
    lines.append("—" * 40)
    lines.append("")
    lines.append(lyrics)
    lines.append("")
    lines.append("—" * 40)

    listeners = context.get("listeners", "")
    if listeners:
        try:
            lines.append("")
            lines.append(f"Heard by {int(listeners):,} people.")
        except Exception:
            pass

    return "\n".join(lines)


def main():
    title, artist = read_request()
    print(f"Requesting lyrics: {title} by {artist}")

    # Try primary source
    raw = get_lyrics_ovh(title, artist)

    # Try fallback
    if not raw:
        print("lyrics.ovh returned nothing — trying chartlyrics...")
        raw = get_lyrics_chartlyrics(title, artist)

    # Pull context regardless of lyrics success
    print("Fetching song context from Last.fm...")
    context = get_song_context(title, artist)

    if raw:
        lyrics = clean_lyrics(raw)
        response = format_response(title, artist, lyrics, context)
    else:
        album = context.get("album", {}).get("title", "")
        tags_raw = context.get("toptags", {}).get("tag", [])
        tags = [t["name"] for t in tags_raw[:5]] if tags_raw else []

        response = f"{artist} — {title}\n"
        if album:
            response += f"  from {album}\n"
        if tags:
            response += f"  tagged: {', '.join(tags)}\n"
        response += (
            f"\nLyrics not found for this track.\n\n"
            f"Both lyrics.ovh and chartlyrics were tried. "
            f"This sometimes happens with less common songs, alternate title spellings, "
            f"or tracks that haven't been indexed yet.\n\n"
            f"Try checking the spelling, or use the most widely known version of the title."
        )

    with open("Audio/lyrics-response.txt", "w") as f:
        f.write(response)

    print("Response written to Audio/lyrics-response.txt")
    print("---")
    print(response[:500])


if __name__ == "__main__":
    main()
