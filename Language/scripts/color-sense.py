import re
import colorsys

COLOR_KNOWLEDGE = {
    "red": {
        "name": "Red",
        "hex": "#FF0000",
        "family": "warm",
        "psychology": "Red activates. It raises heart rate, increases adrenaline, and accelerates breathing — measurably, not metaphorically. It is the color of blood, fire, stop signs, and ripe fruit. It signals both danger and desire, often simultaneously.",
        "associations": "Love, danger, urgency, passion, anger, appetite, power, warning, revolution, vitality.",
        "cultures": "In China, red is the color of luck, celebration, and protection — brides wear red, not white. In Western culture, red signals danger or prohibition. In Hinduism, red is associated with Shakti — divine feminine energy. In ancient Rome, red was the color of Mars, god of war. In many African traditions, red is associated with sacrifice and ritual power.",
        "light": "Red light is the longest wavelength visible to humans — it travels furthest through fog and atmosphere. Red is the first color the human eye detects at low light levels.",
        "feel": "Red demands attention before you decide to give it. It is the least neutral color that exists.",
    },
    "blue": {
        "name": "Blue",
        "hex": "#0000FF",
        "family": "cool",
        "psychology": "Blue slows. It lowers heart rate and blood pressure in measurable ways. It is the color of sky and water — the two things that extend beyond human scale. It is trusted more than any other color in most cultures.",
        "associations": "Calm, trust, depth, sadness, distance, authority, intelligence, melancholy, infinite space.",
        "cultures": "In ancient Egypt, blue represented water and the heavens — lapis lazuli was among the most precious materials. In Western culture, blue is associated with stability and trustworthiness, which is why it dominates corporate identity. In Hinduism, Krishna and Vishnu are depicted in blue. In many Indigenous American traditions, blue is the color of the sky and spiritual connection. In European medieval painting, the Virgin Mary's robe was painted in expensive ultramarine — blue as the divine.",
        "light": "Blue light has the shortest wavelength in the visible spectrum. It scatters most in the atmosphere — which is why the sky is blue and why screens keep us awake: they emit blue light in quantities that confuse the brain into thinking it is still daytime.",
        "feel": "Blue recedes. It makes space feel larger and time feel slower. It is the color that most people name as their favorite.",
    },
    "green": {
        "name": "Green",
        "hex": "#00FF00",
        "family": "cool",
        "psychology": "Green is the color the human eye can distinguish the most shades of — a result of millions of years in forests where detecting movement and edibility in foliage mattered enormously. It is restful in a way that other colors are not.",
        "associations": "Nature, growth, health, envy, renewal, calm, poison, safety, abundance, the natural world.",
        "cultures": "In Islam, green is sacred — the color of paradise and the Prophet's banner. In Western culture, green is the color of money, envy, and environmental consciousness. In Celtic tradition, green was associated with the Otherworld and the fairy realm — both sacred and dangerous. In China, green jade is among the most valuable materials. In ancient Egypt, Osiris — god of resurrection — was depicted with green skin.",
        "light": "The human eye has more cone cells sensitive to green than to any other color. We see more shades of green than any other color in the spectrum.",
        "feel": "Green is the color most associated with recovery and restoration. Hospital walls were painted green for decades based on research showing it reduces stress.",
    },
    "black": {
        "name": "Black",
        "hex": "#000000",
        "family": "neutral",
        "psychology": "Black is the absence of visible light — but to the human eye it is not nothing. It is presence, weight, and authority. It creates contrast that makes everything near it more visible. It is simultaneously the color of mourning and elegance.",
        "associations": "Absence, night, death, elegance, power, mystery, formality, protection, the unknown, depth.",
        "cultures": "In Western culture, black is mourning — the absence of color as the absence of life. In ancient Egypt, black was the color of fertile Nile soil — it represented fertility and life, not death. In Japan, black is formal and powerful, associated with ink, calligraphy, and the samurai. In many Indigenous traditions, black is the color of the West — the direction of introspection and the night. In contemporary fashion, black is sophistication: endless, adaptable, and never wrong.",
        "light": "True black absorbs all wavelengths of visible light and reflects none. Vantablack — the darkest substance created — reflects 0.035% of light, and objects painted in it appear to lose their three-dimensional form.",
        "feel": "Black does not recede — it holds. It is the color that makes other colors possible to see.",
    },
    "white": {
        "name": "White",
        "hex": "#FFFFFF",
        "family": "neutral",
        "psychology": "White contains all wavelengths of visible light equally. It reflects rather than absorbs. It is associated with beginnings, cleanliness, and the blank — both the potential of the unmarked page and the blankness of erasure.",
        "associations": "Purity, light, beginning, emptiness, cleanliness, surrender, peace, clinical distance, the infinite.",
        "cultures": "In many Eastern cultures, white is the color of death and mourning — widows wear white. In Western culture, white is worn by brides — purity and beginning. In ancient Rome, white togas signified citizenship. In many Indigenous traditions, white is the color of the North — clarity, wisdom, winter. In Islamic tradition, white is associated with purity and is worn for pilgrimage.",
        "light": "White light contains the full visible spectrum. A prism reveals this by splitting it into its component colors. Snow appears white because ice crystals scatter and reflect all wavelengths equally.",
        "feel": "White space in design is not emptiness — it is breathing room. White makes other things legible.",
    },
    "gold": {
        "name": "Gold",
        "hex": "#FFD700",
        "family": "warm",
        "psychology": "Gold is associated with worth — not only financial but spiritual. It is the color of the sun at specific moments: late afternoon, winter, the hour before dark. It promises something that silver does not.",
        "associations": "Value, achievement, warmth, the divine, success, excess, harvest, transformation, memory.",
        "cultures": "In virtually every culture with access to the metal, gold has been considered sacred. In ancient Egypt, gold was the flesh of the gods. In the Catholic tradition, gold leaf in paintings represents divine light — not decoration but theology. In many Indigenous traditions, gold ochre is one of the oldest pigments used in ceremony. In Aztec culture, gold was not currency but sacred material — Tenochtitlan was so gold it blinded the Spanish.",
        "light": "The warm golden light of the hour before sunset — photographers call it the golden hour — is the moment when the sun's angle through the atmosphere filters out blue wavelengths, leaving only red, orange, and gold.",
        "feel": "Gold is warm the way a fire is warm — it draws the eye and suggests safety and abundance in a way cooler colors cannot.",
    },
    "purple": {
        "name": "Purple",
        "hex": "#800080",
        "family": "cool",
        "psychology": "Purple sits at the edge of the visible spectrum — beyond blue, approaching ultraviolet. It is the color associated with royalty for a specific historical reason: Tyrian purple dye required thousands of sea snails to produce a small amount, making it more expensive than gold.",
        "associations": "Royalty, spirituality, mystery, creativity, transformation, the liminal, magic, mourning, extravagance.",
        "cultures": "In Rome, only the emperor could wear Tyrian purple — its price made it a symbol of absolute power. In Byzantine Christianity, purple was imperial and sacred. In many Buddhist traditions, purple is associated with spiritual transformation. In Western culture, purple mourning was used for centuries in royal and aristocratic families. In the queer community, purple has been reclaimed as a color of identity and pride.",
        "light": "Purple is not a spectral color — there is no single wavelength of light that appears purple. The brain creates it by mixing red and blue signals. It is, in a precise sense, something the eye invents.",
        "feel": "Purple is the color most associated with the space between — between day and night, between the ordinary and the sacred, between one state and another.",
    },
    "orange": {
        "name": "Orange",
        "hex": "#FFA500",
        "family": "warm",
        "psychology": "Orange is the most energetic warm color that does not carry red's aggression. It is associated with heat, ripeness, and the specific quality of fire at its warmest. It is the color of autumn — things at the peak of their transformation before release.",
        "associations": "Energy, warmth, harvest, creativity, enthusiasm, vitality, the transitional, caution, appetite.",
        "cultures": "In Buddhism and Hinduism, orange-saffron robes represent renunciation, fire, and spiritual seeking. In Western culture, orange is the color of Halloween — the harvest season, the boundary between living and dead. In the Netherlands, orange is a national color — worn to celebrate Dutch identity. In many Indigenous traditions of the American Southwest, orange ochre is a sacred pigment used in ceremony and healing.",
        "light": "Orange light is the color of fire and of the sun near the horizon — when the atmosphere scatters blue wavelengths and the long red-orange ones remain. Candlelight is orange. Firelight is orange.",
        "feel": "Orange warms a space in a way that yellow doesn't quite manage. It is the color of things being used — of fire doing its work, of fruit at its peak.",
    },
    "grey": {
        "name": "Grey",
        "hex": "#808080",
        "family": "neutral",
        "psychology": "Grey is the color of neither/nor — neither black nor white, neither one thing nor another. It is the color of overcast days, of concrete, of ash, of fog. It is often associated with neutrality and balance, but also with depression, in climates where grey sky is the dominant experience for months.",
        "associations": "Neutrality, fog, ambiguity, winter, age, ash, stone, transition, the in-between, quietness.",
        "cultures": "In Japanese aesthetics, grey — especially grey that approaches white — is associated with wabi-sabi: the beauty of impermanence and incompleteness. In many Western contexts, grey is the color of institutional spaces. In Norse mythology, fog and grey are associated with the world between worlds. In East Asian ink painting, grey — the dilution of black ink — carries as much meaning as pure black.",
        "light": "Grey light is diffuse light — when cloud cover scatters the sun's rays so evenly that no shadow is cast. Photographers call it flat light. It reveals texture where directional light would create shadow.",
        "feel": "Grey is the color of the held breath. It is not empty — it is waiting.",
    },
    "pink": {
        "name": "Pink",
        "hex": "#FFC0CB",
        "family": "warm",
        "psychology": "Pink is red's softened form — the warmth of red without its urgency. Studies have shown that bright pink suppresses aggression in confined populations, enough that sports teams have painted opposing locker rooms pink. Dusty or muted pinks carry a different register entirely — nostalgia, tenderness, fading.",
        "associations": "Tenderness, romance, youth, innocence, beauty, the body, softness, care, nostalgia, the gentle.",
        "cultures": "Pink's association with femininity is recent and specific to Western culture — in the early 20th century, pink was considered a boy's color (a pale version of red, which was considered masculine). In Japan, pink is associated with cherry blossoms — impermanence, fleeting beauty, and spring. In Sufi poetry, the pink rose is associated with divine love. In drag and queer culture, pink has been reclaimed as a symbol of visible identity.",
        "light": "Pink light occurs at sunrise and sunset when the atmosphere scatters blue light and the red wavelengths mix with the white reflected from clouds. It is the light that makes everything look briefly more beautiful than usual.",
        "feel": "Pink softens. It is the color most associated with the sensation of being taken care of.",
    },
}

ALIASES = {
    "gray": "grey",
    "crimson": "red",
    "scarlet": "red",
    "burgundy": "red",
    "navy": "blue",
    "indigo": "blue",
    "cyan": "blue",
    "teal": "blue",
    "cerulean": "blue",
    "olive": "green",
    "lime": "green",
    "emerald": "green",
    "forest": "green",
    "sage": "green",
    "ebony": "black",
    "obsidian": "black",
    "ivory": "white",
    "cream": "white",
    "silver": "grey",
    "charcoal": "grey",
    "slate": "grey",
    "amber": "orange",
    "rust": "orange",
    "copper": "orange",
    "lavender": "purple",
    "violet": "purple",
    "mauve": "purple",
    "lilac": "purple",
    "rose": "pink",
    "blush": "pink",
    "salmon": "pink",
    "magenta": "pink",
    "golden": "gold",
    "yellow": "gold",
    "saffron": "gold",
    "ochre": "gold",
}


def read_request():
    with open("Language/color-request.txt", "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        raise ValueError("color-request.txt is empty.")
    return lines[0].strip()


def parse_hex(query):
    query = query.strip().lstrip('#')
    if re.match(r'^[0-9a-fA-F]{6}$', query) or re.match(r'^[0-9a-fA-F]{3}$', query):
        if len(query) == 3:
            query = ''.join(c*2 for c in query)
        r = int(query[0:2], 16) / 255
        g = int(query[2:4], 16) / 255
        b = int(query[4:6], 16) / 255
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return r, g, b, h, s, v, '#' + query.upper()
    return None


def find_nearest_named(h, s, v):
    if v < 0.15:
        return "black"
    if v > 0.85 and s < 0.15:
        return "white"
    if s < 0.15:
        return "grey"
    h_deg = h * 360
    if h_deg < 15 or h_deg >= 345:
        return "red"
    elif h_deg < 45:
        return "orange"
    elif h_deg < 75:
        return "gold"
    elif h_deg < 165:
        return "green"
    elif h_deg < 255:
        return "blue"
    elif h_deg < 285:
        return "purple"
    elif h_deg < 330:
        return "pink"
    else:
        return "red"


def find_color(query):
    q = query.lower().strip()
    # Try hex
    hex_result = parse_hex(q)
    if hex_result:
        r, g, b, h, s, v, hex_str = hex_result
        nearest = find_nearest_named(h, s, v)
        return nearest, COLOR_KNOWLEDGE.get(nearest), hex_str
    # Direct name match
    if q in COLOR_KNOWLEDGE:
        return q, COLOR_KNOWLEDGE[q], None
    # Alias match
    if q in ALIASES:
        key = ALIASES[q]
        return key, COLOR_KNOWLEDGE[key], None
    # Partial match
    for key in COLOR_KNOWLEDGE:
        if key in q or q in key:
            return key, COLOR_KNOWLEDGE[key], None
    return None, None, None


def format_response(query, key, knowledge, hex_override=None):
    lines = []
    display_name = query.title() if query.lower() != key else knowledge["name"]
    if hex_override:
        lines.append(f"{display_name}  →  {knowledge['name']}  {hex_override}")
    else:
        lines.append(f"{display_name}  ·  {knowledge['hex']}")
    lines.append(f"  {knowledge['family'].title()} spectrum")
    lines.append("")
    lines.append(knowledge["psychology"])
    lines.append("")
    lines.append(f"Associated with: {knowledge['associations']}")
    lines.append("")
    lines.append("—" * 40)
    lines.append("")
    lines.append("Across cultures:")
    lines.append(knowledge["cultures"])
    lines.append("")
    lines.append("Light:")
    lines.append(knowledge["light"])
    lines.append("")
    lines.append("—" * 40)
    lines.append("")
    lines.append(knowledge["feel"])
    return "\n".join(lines)


def format_unknown(query):
    lines = []
    lines.append(f"Color: {query.title()}")
    lines.append("")
    lines.append("This color isn't in the built-in knowledge base.")
    lines.append("")
    lines.append("You can also input a hex code (e.g. #4a2060) and the nearest named color will be returned.")
    lines.append("")
    lines.append("Known colors: " + ", ".join(COLOR_KNOWLEDGE.keys()))
    lines.append("Understood aliases: " + ", ".join(sorted(ALIASES.keys())))
    return "\n".join(lines)


def main():
    query = read_request()
    print(f"Looking up color: {query}")

    key, knowledge, hex_override = find_color(query)

    if knowledge:
        response = format_response(query, key, knowledge, hex_override)
    else:
        response = format_unknown(query)

    with open("Language/color-response.txt", "w") as f:
        f.write(response)

    print("Response written to Language/color-response.txt")
    print("---")
    print(response[:400])


if __name__ == "__main__":
    main()
