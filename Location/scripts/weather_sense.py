import requests
import datetime

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


def read_request():
    with open("Location/weather-request.txt", "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        raise ValueError("weather-request.txt is empty. Write a place name.")
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
    }


def get_weather(lat, lon, timezone):
    r = requests.get(WEATHER_URL, params={
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_gusts_10m",
            "wind_direction_10m",
            "weather_code",
            "cloud_cover",
            "precipitation",
            "visibility",
            "uv_index",
            "is_day",
            "surface_pressure",
        ],
        "hourly": [
            "temperature_2m",
            "apparent_temperature",
            "weather_code",
            "precipitation_probability",
        ],
        "daily": [
            "sunrise",
            "sunset",
            "temperature_2m_max",
            "temperature_2m_min",
            "uv_index_max",
            "precipitation_sum",
            "wind_speed_10m_max",
        ],
        "timezone": timezone,
        "wind_speed_unit": "mph",
        "temperature_unit": "celsius",
        "forecast_days": 1,
    })
    r.raise_for_status()
    return r.json()


def wind_direction_name(degrees):
    if degrees is None:
        return ""
    directions = [
        "north", "north-northeast", "northeast", "east-northeast",
        "east", "east-southeast", "southeast", "south-southeast",
        "south", "south-southwest", "southwest", "west-southwest",
        "west", "west-northwest", "northwest", "north-northwest"
    ]
    idx = round(degrees / 22.5) % 16
    return directions[idx]


def describe_temperature(temp_c, feels_c):
    diff = feels_c - temp_c
    feel = feels_c

    if feel <= -20:
        base = "brutal cold — the kind that makes metal dangerous to touch"
    elif feel <= -10:
        base = "deep cold, the kind that gets inside your coat"
    elif feel <= 0:
        base = "freezing — breath visible, surfaces hard underfoot"
    elif feel <= 5:
        base = "raw cold, the kind that finds exposed skin"
    elif feel <= 10:
        base = "cold enough to feel alive in it"
    elif feel <= 15:
        base = "cool — a cardigan kind of temperature"
    elif feel <= 20:
        base = "mild, neither pulling toward warmth nor away from it"
    elif feel <= 25:
        base = "warm — comfortable in light clothes, the air soft"
    elif feel <= 30:
        base = "properly warm, the kind that slows you down pleasantly"
    elif feel <= 35:
        base = "hot — the air has weight to it"
    else:
        base = "intense heat, the kind that bends the light off surfaces"

    if diff <= -5:
        wind_note = " Wind is pulling heat from it — feels colder than it reads."
    elif diff >= 5:
        wind_note = " Humidity is holding the heat in — feels heavier than it reads."
    else:
        wind_note = ""

    return f"{feel:.0f}°C ({temp_c:.0f}°C actual) — {base}.{wind_note}"


def describe_humidity(humidity):
    if humidity < 20:
        return "very dry — the kind of air that makes lips crack and static spark"
    elif humidity < 40:
        return "dry and clear, easy to breathe"
    elif humidity < 60:
        return "comfortable — the air is neutral, neither wet nor parched"
    elif humidity < 75:
        return "noticeable moisture in the air, clothes feel slightly heavier"
    elif humidity < 90:
        return "humid — the air is thick, everything takes longer to dry"
    else:
        return "saturated — the air is almost water, warm and very close"


def describe_wind(speed, gusts, direction_deg):
    direction = wind_direction_name(direction_deg)
    from_dir = f"from the {direction}, " if direction else ""

    if speed < 5:
        base = "almost still — barely a movement in leaves"
    elif speed < 15:
        base = "a light breeze, something you feel on your face but don't fight"
    elif speed < 25:
        base = "steady wind — hair moves, loose things shift"
    elif speed < 40:
        base = "strong wind, the kind that requires leaning into"
    else:
        base = "very strong wind — difficult to stand still against"

    gust_note = ""
    if gusts and gusts > speed + 15:
        gust_note = f" Gusting to {gusts:.0f}mph."

    return f"{from_dir}{base} ({speed:.0f}mph){gust_note}"


def describe_sky(cloud_cover, weather_code, is_day):
    time_of_day = "day" if is_day else "night"

    if weather_code == 0:
        sky = "completely clear" if is_day else "open sky, stars visible"
    elif weather_code in (1, 2):
        sky = "mostly clear with some cloud" if is_day else "partly cloudy"
    elif weather_code == 3:
        sky = "overcast — a uniform grey ceiling"
    elif weather_code in (45, 48):
        sky = "fog — visibility low, the world compressed to what's close"
    elif weather_code in (51, 53, 55):
        sky = "drizzle — light and persistent, more felt than seen"
    elif weather_code in (61, 63, 65):
        sky = "rain — steady and real"
    elif weather_code in (71, 73, 75, 77):
        sky = "snow — the particular silence it brings"
    elif weather_code in (80, 81, 82):
        sky = "rain showers — coming and going"
    elif weather_code in (85, 86):
        sky = "snow showers"
    elif weather_code in (95, 96, 99):
        sky = "thunderstorm — pressure and electricity in the air"
    else:
        if cloud_cover < 20:
            sky = "clear" if is_day else "open night sky"
        elif cloud_cover < 60:
            sky = "partly cloudy"
        else:
            sky = "overcast"

    return f"{sky} ({time_of_day})"


def describe_visibility(vis_m):
    if vis_m is None:
        return None
    vis_km = vis_m / 1000
    if vis_km >= 20:
        return "excellent — distant things are sharp"
    elif vis_km >= 10:
        return "good — clear to the horizon"
    elif vis_km >= 5:
        return "moderate — some haze at distance"
    elif vis_km >= 1:
        return "poor — the middle distance is blurred"
    else:
        return "very poor — close visibility only, fog or heavy precipitation"


def describe_uv(uv):
    if uv is None:
        return None
    if uv < 1:
        return "none"
    elif uv < 3:
        return "low"
    elif uv < 6:
        return "moderate — skin will feel it after a while"
    elif uv < 8:
        return "high — unprotected skin burns faster than expected"
    elif uv < 11:
        return "very high — limit direct exposure"
    else:
        return "extreme"


def format_hourly_outlook(data, timezone):
    """Pull the next 6 hours of temperature and conditions."""
    times = data.get("hourly", {}).get("time", [])
    temps = data.get("hourly", {}).get("temperature_2m", [])
    codes = data.get("hourly", {}).get("weather_code", [])
    precip_prob = data.get("hourly", {}).get("precipitation_probability", [])

    now_str = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:00")
    try:
        start_idx = times.index(now_str)
    except ValueError:
        # Find closest
        start_idx = 0

    outlook = []
    for i in range(start_idx, min(start_idx + 6, len(times))):
        hour = times[i][11:16]  # HH:MM
        temp = temps[i] if i < len(temps) else None
        code = codes[i] if i < len(codes) else 0
        pp = precip_prob[i] if i < len(precip_prob) else None

        condition = describe_sky(50, code, 1).split(" (")[0]
        line = f"  {hour}: {temp:.0f}°C, {condition}"
        if pp and pp > 30:
            line += f" ({pp}% chance of rain)"
        outlook.append(line)

    return outlook


def sensory_synthesis(feels, humidity, code, is_day, cloud, precip, wind_speed, uv):
    """Generate a closing sensory paragraph."""
    notes = []

    if feels <= 0 and code in (71, 73, 75, 77):
        notes.append("Very quiet right now. Snow absorbs sound — the world is muffled and close.")
    elif feels >= 28 and humidity >= 70:
        notes.append("Heat and moisture together — the air doesn't move so much as press.")
    elif code in (95, 96, 99):
        notes.append("There is electricity in it. The kind of weather that demands attention.")
    elif code == 0 and is_day and feels >= 18:
        if uv and uv >= 6:
            notes.append("Clear and genuinely hot. The sun has teeth today.")
        else:
            notes.append("Clear and warm. The kind of day that asks you to be outside in it.")
    elif code == 0 and not is_day:
        notes.append("Clear night. The kind of dark that has depth to it.")
    elif code in (45, 48):
        notes.append("Fog closes the world to what is immediate. Everything else disappears.")
    elif feels <= -10:
        notes.append("This is serious cold. It requires preparation and respect.")
    elif feels <= 5:
        notes.append("Cold that makes you aware of your own warmth.")
    elif cloud >= 80 and precip == 0:
        notes.append("The kind of grey that is neither threatening nor beautiful. Just present.")
    elif code in (51, 53, 55):
        notes.append("Drizzle — the kind of wet that accumulates without drama.")
    elif code in (61, 63, 65):
        notes.append("Real rain. The kind that changes your plans.")
    elif feels >= 20 and cloud < 40:
        notes.append("Pleasant. The kind of weather that disappears into the background of a good day.")

    if wind_speed >= 40:
        notes.append("The wind is the dominant fact of this weather.")
    elif wind_speed >= 25 and not notes:
        notes.append("The wind makes itself known — it has opinions about where you stand.")

    return notes


def format_response(place_info, data):
    name = place_info["name"]
    country = place_info["country"]
    admin = place_info.get("admin1", "")

    parts = [name]
    if admin and admin != name:
        parts.append(admin)
    if country:
        parts.append(country)
    label = ", ".join(parts)

    current = data.get("current", {})
    daily = data.get("daily", {})

    temp = current.get("temperature_2m", 0)
    feels = current.get("apparent_temperature", temp)
    humidity = current.get("relative_humidity_2m", 50)
    wind = current.get("wind_speed_10m", 0)
    gusts = current.get("wind_gusts_10m", wind)
    wind_dir = current.get("wind_direction_10m")
    cloud = current.get("cloud_cover", 0)
    code = current.get("weather_code", 0)
    is_day = current.get("is_day", 1)
    precip = current.get("precipitation", 0)
    visibility = current.get("visibility")
    uv = current.get("uv_index")

    sunrise = daily.get("sunrise", [None])[0]
    sunset = daily.get("sunset", [None])[0]
    temp_max = daily.get("temperature_2m_max", [None])[0]
    temp_min = daily.get("temperature_2m_min", [None])[0]
    precip_sum = daily.get("precipitation_sum", [None])[0]

    lines = []
    lines.append(label)
    lines.append(f"  {datetime.datetime.utcnow().strftime('%A, %B %d — %H:%M UTC')}")
    lines.append("")
    lines.append(describe_temperature(temp, feels))
    if temp_max is not None and temp_min is not None:
        lines.append(f"  Today's range: {temp_min:.0f}°C to {temp_max:.0f}°C")
    lines.append("")
    lines.append(f"Sky: {describe_sky(cloud, code, is_day)}")
    lines.append(f"Air: {describe_humidity(humidity)}")
    lines.append(f"Wind: {describe_wind(wind, gusts, wind_dir)}")

    vis_desc = describe_visibility(visibility)
    if vis_desc:
        lines.append(f"Visibility: {vis_desc}")

    uv_desc = describe_uv(uv)
    if uv_desc and is_day:
        lines.append(f"UV: {uv_desc}")

    if precip > 0:
        lines.append(f"Precipitation: {precip}mm falling right now")

    if sunrise and sunset:
        sr = sunrise[11:16] if len(sunrise) > 10 else sunrise
        ss = sunset[11:16] if len(sunset) > 10 else sunset
        lines.append(f"Sun: rises {sr}, sets {ss} (local time)")

    # Hourly outlook
    hourly = format_hourly_outlook(data, place_info["timezone"])
    if hourly:
        lines.append("")
        lines.append("Next few hours:")
        lines.extend(hourly)

    # Sensory closing
    notes = sensory_synthesis(feels, humidity, code, is_day, cloud, precip, wind, uv)
    if notes:
        lines.append("")
        for note in notes:
            lines.append(note)

    return "\n".join(lines)


def main():
    place_query = read_request()
    print(f"Looking up: {place_query}")

    try:
        place_info = geocode(place_query)
        print(f"Found: {place_info['name']}, {place_info['country']}")
        data = get_weather(place_info["latitude"], place_info["longitude"], place_info["timezone"])
        response = format_response(place_info, data)
    except ValueError as e:
        response = str(e)

    with open("Location/weather-response.txt", "w") as f:
        f.write(response)

    print("Response written to Location/weather-response.txt")
    print("---")
    print(response)


if __name__ == "__main__":
    main()
