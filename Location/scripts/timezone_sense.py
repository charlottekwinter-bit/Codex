import requests
import datetime
import math

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"


def read_request():
    with open("Location/timezone-request.txt", "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        raise ValueError("timezone-request.txt is empty. Write a place name.")
    return " ".join(lines)


def geocode(place):
    r = requests.get(GEOCODE_URL, params={
        "name": place,
        "count": 1,
        "language": "en",
        "format": "json"
    })
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        raise ValueError(f"Could not find a location for: {place}")
    result = results[0]
    return {
        "name": result.get("name", place),
        "country": result.get("country", ""),
        "admin1": result.get("admin1", ""),
        "latitude": result["latitude"],
        "longitude": result["longitude"],
        "timezone": result.get("timezone", "UTC"),
        "utc_offset_seconds": result.get("utc_offset_seconds", 0),
    }


def get_local_time(utc_offset_seconds):
    utc_now = datetime.datetime.utcnow()
    offset = datetime.timedelta(seconds=utc_offset_seconds)
    local_time = utc_now + offset
    return local_time


def describe_hour(hour, minute):
    """Return a sensory description of this moment in the day."""
    if 0 <= hour < 4:
        period = "deep night"
        feel = "The city is at its most itself right now — the part that doesn't perform. Almost everyone is asleep. The streets belong to a different set of people."
    elif hour < 6:
        period = "pre-dawn"
        feel = "The last hour before the sky changes. Bakeries and hospitals and night workers. The quiet before the quiet ends."
    elif hour < 8:
        period = "early morning"
        feel = "The day is beginning. Coffee, commutes, the sound of the city finding its rhythm. Light arriving at a low angle."
    elif hour < 10:
        period = "morning"
        feel = "The day in full motion. Everything open, everyone moving. The particular energy of hours that still feel full of possibility."
    elif hour < 12:
        period = "late morning"
        feel = "The productive middle of the morning. The day no longer feels new but isn't yet half gone."
    elif hour < 14:
        period = "midday"
        feel = "The sun at or near its highest. Lunch, a pause in the rhythm, the particular light of noon. In hot places, the beginning of the slow afternoon."
    elif hour < 16:
        period = "early afternoon"
        feel = "The afternoon stretch. Energy dips here in most humans — the biology of the post-lunch hours. In warm places, the siesta hour."
    elif hour < 18:
        period = "late afternoon"
        feel = "The light is shifting. The day tilting toward evening. The particular quality of afternoon light — golden, low, long shadows."
    elif hour < 20:
        period = "evening"
        feel = "The transition hour. Work ending, dinner beginning, the city changing its tone. The light going warm and then gone."
    elif hour < 22:
        period = "night"
        feel = "The evening in full. Restaurants, families at dinner, the city in its social hour. The sky fully dark."
    else:
        period = "late night"
        feel = "The night thinning out. Most people moving toward sleep. The city quieter now, the streets more spacious."

    return period, feel


def describe_offset(offset_seconds):
    hours = offset_seconds / 3600
    if hours == 0:
        return "UTC (no offset)"
    elif hours > 0:
        return f"UTC+{hours:.1f}".rstrip('0').rstrip('.')
    else:
        return f"UTC{hours:.1f}".rstrip('0').rstrip('.')


def format_response(place_info, local_time):
    name = place_info["name"]
    country = place_info["country"]
    admin = place_info.get("admin1", "")
    timezone = place_info["timezone"]
    offset = place_info["utc_offset_seconds"]

    parts = [name]
    if admin and admin != name:
        parts.append(admin)
    if country:
        parts.append(country)
    label = ", ".join(parts)

    hour = local_time.hour
    minute = local_time.minute
    period, feel = describe_hour(hour, minute)

    time_str = local_time.strftime("%H:%M")
    day_str = local_time.strftime("%A, %B %d")
    offset_str = describe_offset(offset)

    lines = []
    lines.append(label)
    lines.append("")
    lines.append(f"It is {time_str} on {day_str}.")
    lines.append(f"  {timezone} · {offset_str}")
    lines.append("")
    lines.append(f"That is: {period}.")
    lines.append("")
    lines.append(feel)

    # Difference from UTC
    utc_now = datetime.datetime.utcnow()
    utc_str = utc_now.strftime("%H:%M")
    diff_hours = offset / 3600
    if diff_hours != 0:
        direction = "ahead of" if diff_hours > 0 else "behind"
        abs_diff = abs(diff_hours)
        diff_str = f"{abs_diff:.1f}".rstrip('0').rstrip('.')
        lines.append("")
        lines.append(f"{name} is {diff_str} hour{'s' if abs_diff != 1 else ''} {direction} UTC. Right now in UTC it is {utc_str}.")

    return "\n".join(lines)


def main():
    place_query = read_request()
    print(f"Looking up timezone for: {place_query}")

    try:
        place_info = geocode(place_query)
        local_time = get_local_time(place_info["utc_offset_seconds"])
        response = format_response(place_info, local_time)
    except ValueError as e:
        response = str(e)

    with open("Location/timezone-response.txt", "w") as f:
        f.write(response)

    print("Response written to Location/timezone-response.txt")
    print("---")
    print(response)


if __name__ == "__main__":
    main()
