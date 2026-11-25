"""Deterministic in-memory FastMCP simulators for the six travel servers.

They expose the *canonical* tool contract the planner is taught. Every response is a pure function
of ``(server, tool, args)`` (seeded by SHA-256), so experiments are reproducible offline.
Results produced here are always tagged ``source=sim`` by the proxy - never reported as live data.
"""
from __future__ import annotations

import hashlib
import json
import math
import datetime as _dt
import random

from fastmcp import FastMCP

PLACES: dict[str, tuple[float, float, str]] = {
    # airports
    "MAN": (53.3537, -2.2750, "GB"), "LHR": (51.4700, -0.4543, "GB"), "EDI": (55.9500, -3.3725, "GB"),
    "BCN": (41.2974, 2.0833, "ES"), "LIS": (38.7742, -9.1342, "PT"), "FCO": (41.8003, 12.2389, "IT"),
    "AMS": (52.3105, 4.7683, "NL"), "PRG": (50.1008, 14.2600, "CZ"), "VIE": (48.1103, 16.5697, "AT"),
    "CPH": (55.6180, 12.6508, "DK"), "PMI": (39.5517, 2.7388, "ES"), "ATH": (37.9364, 23.9445, "GR"),
    "NAP": (40.8860, 14.2908, "IT"), "SPU": (43.5389, 16.2980, "HR"),
    # ports
    "Piraeus": (37.9420, 23.6465, "GR"), "Naples": (40.8378, 14.2551, "IT"), "Split": (43.5040, 16.4400, "HR"),
    # cities
    "Manchester": (53.4808, -2.2426, "GB"), "London": (51.5072, -0.1276, "GB"),
    "Edinburgh": (55.9533, -3.1883, "GB"), "Barcelona": (41.3874, 2.1686, "ES"),
    "Lisbon": (38.7223, -9.1393, "PT"), "Rome": (41.9028, 12.4964, "IT"), "Amsterdam": (52.3676, 4.9041, "NL"),
    "Prague": (50.0755, 14.4378, "CZ"), "Vienna": (48.2082, 16.3738, "AT"),
    "Copenhagen": (55.6761, 12.5683, "DK"), "Palma": (39.5696, 2.6502, "ES"),
    "Naxos": (37.1036, 25.3766, "GR"), "Capri": (40.5532, 14.2222, "IT"), "Hvar": (43.1729, 16.4411, "HR"),
    # added for data/tasks_100.json: airports
    "BHX": (52.4539, -1.7480, "GB"), "BRS": (51.3827, -2.7191, "GB"), "BER": (52.3667, 13.5033, "DE"),
    "CDG": (49.0097, 2.5479, "FR"), "MAD": (40.4983, -3.5676, "ES"), "OPO": (41.2481, -8.6814, "PT"),
    "MXP": (45.6301, 8.7255, "IT"), "DUB": (53.4264, -6.2499, "IE"),
    # added for data/tasks_100.json: cities
    "Birmingham": (52.4862, -1.8904, "GB"), "Bristol": (51.4545, -2.5879, "GB"), "Berlin": (52.5200, 13.4050, "DE"),
    "Paris": (48.8566, 2.3522, "FR"), "Madrid": (40.4168, -3.7038, "ES"), "Athens": (37.9838, 23.7275, "GR"),
    "Porto": (41.1579, -8.6291, "PT"), "Milan": (45.4642, 9.1900, "IT"), "Dublin": (53.3498, -6.2603, "IE"),
    "Santorini": (36.3932, 25.4615, "GR"), "Mykonos": (37.4467, 25.3289, "GR"), "Paros": (37.0853, 25.1522, "GR"),
    "Ibiza": (38.9067, 1.4206, "ES"), "Menorca": (39.9496, 4.1108, "ES"),
}
CITY_COST = {"Copenhagen": 1.5, "Amsterdam": 1.45, "Capri": 1.7, "Naxos": 1.0, "Hvar": 1.1,
             "Barcelona": 1.2, "Rome": 1.2, "Lisbon": 1.0, "Prague": 0.8, "Vienna": 1.1, "Palma": 1.15,
             "Berlin": 1.0, "Paris": 1.45, "Madrid": 1.1, "Athens": 1.0, "Porto": 0.9, "Milan": 1.3, "Dublin": 1.4,
             "Naples": 1.0, "Split": 1.0, "Santorini": 1.5, "Mykonos": 1.6, "Paros": 1.1, "Ibiza": 1.4,
             "Menorca": 1.15}
FERRY_ROUTES = {("Piraeus", "Naxos"): (45.0, 230), ("Naples", "Capri"): (24.0, 50), ("Split", "Hvar"): (18.0, 65),
                ("Piraeus", "Santorini"): (50.0, 320), ("Piraeus", "Mykonos"): (48.0, 230),
                ("Piraeus", "Paros"): (42.0, 210), ("Barcelona", "Ibiza"): (55.0, 480),
                ("Barcelona", "Menorca"): (55.0, 480)}
CARRIERS = ["BA", "U2", "FR", "VY", "TP", "AZ", "KL", "OS", "SK", "A3", "W6"]
HOTEL_WORDS = ["Grand", "Harbour", "Old Town", "Boutique", "Plaza", "Garden", "Marina", "Central", "Royal", "Loft"]
CATEGORIES = ["food", "culture", "outdoor", "nightlife"]


def _rng(*parts) -> random.Random:
    h = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def haversine_km(a: str, b: str) -> float:
    (la1, lo1, _), (la2, lo2, _) = PLACES[a], PLACES[b]
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def _season(d: _dt.date) -> float:
    return {6: 1.35, 7: 1.4, 8: 1.4, 5: 1.15, 9: 1.15}.get(d.month, 1.0)


# ---------------------------------------------------------------------------------- flights
def build_flights() -> FastMCP:
    mcp = FastMCP("sim-flights")

    @mcp.tool
    def search_flights(origin: str, destination: str, date: str, max_stops: int | None = None) -> dict:
        """Search one-way flight offers. Prices are per passenger in GBP."""
        if origin not in PLACES or destination not in PLACES:
            return {"origin": origin, "destination": destination, "date": date, "offers": []}
        d = _dt.date.fromisoformat(date)
        km = haversine_km(origin, destination)
        base = (40 + 0.085 * km) * _season(d) * (1.12 if d.weekday() >= 4 else 1.0)
        rng = _rng("flights", origin, destination, date)
        has_direct = km < 1500 or _rng("direct", origin, destination).random() < 0.6
        stops_pool = [0, 0, 1, 1, 1, 2] if has_direct else [1, 1, 1, 2, 2, 1]
        offers = []
        for i, stops in enumerate(stops_pool):
            price = base * (1 - 0.12 * stops) * rng.uniform(0.85, 1.25)
            carrier = rng.choice(CARRIERS)
            offers.append({
                "id": f"{origin}{destination}-{date}-{i}", "carrier": carrier,
                "flight_no": f"{carrier}{rng.randint(100, 9999)}", "from": origin, "to": destination,
                "date": date, "depart_time": f"{rng.randint(6, 21):02d}:{rng.choice(['00', '15', '30', '45'])}",
                "stops": stops, "duration_min": int(km / 13 + 40 + stops * rng.randint(70, 180)),
                "price": round(price, 2),
            })
        if max_stops is not None:
            offers = [o for o in offers if o["stops"] <= max_stops]
        return {"origin": origin, "destination": destination, "date": date, "offers": offers}

    return mcp


# ---------------------------------------------------------------------------------- hotels
def build_hotels() -> FastMCP:
    mcp = FastMCP("sim-hotels")

    @mcp.tool
    def search_hotels(city: str, checkin: str, checkout: str, min_rating: float | None = None) -> dict:
        """Search hotels. price_per_night is per room in GBP (may be null; then use total_price)."""
        ci, co = _dt.date.fromisoformat(checkin), _dt.date.fromisoformat(checkout)
        nights = max((co - ci).days, 0)
        rng = _rng("hotels", city, checkin, checkout)
        idx = CITY_COST.get(city, 1.0)
        hotels = []
        for i in range(8):
            rating = round(rng.uniform(3.0, 4.9), 1)
            ppn = round(idx * (55 + 48 * (rating - 3.0)) * _season(ci) * rng.uniform(0.85, 1.2), 2)
            hotels.append({
                "id": f"H-{city[:3].upper()}-{i}", "name": f"{rng.choice(HOTEL_WORDS)} {city} {i + 1}",
                "city": city, "rating": rating,
                "price_per_night": None if rng.random() < 0.12 else ppn,
                "total_price": round(ppn * nights, 2), "refundable": rng.random() < 0.55,
                "checkin": checkin, "checkout": checkout,
            })
        if min_rating is not None:
            hotels = [h for h in hotels if h["rating"] >= min_rating]
        return {"city": city, "checkin": checkin, "checkout": checkout, "nights": nights, "hotels": hotels}

    return mcp


# ---------------------------------------------------------------------------------- geo
def build_geo() -> FastMCP:
    mcp = FastMCP("sim-geo")

    @mcp.tool
    def geocode(query: str) -> dict:
        """Geocode a city, port or IATA airport code."""
        if query not in PLACES:
            return {"query": query, "found": False}
        lat, lon, cc = PLACES[query]
        return {"query": query, "found": True, "name": query, "lat": lat, "lon": lon, "country": cc}

    @mcp.tool
    def directions(from_place: str, to_place: str, profile: str = "driving") -> dict:
        """Road/transit directions between two known places."""
        if from_place not in PLACES or to_place not in PLACES:
            return {"from_place": from_place, "to_place": to_place, "found": False}
        km = haversine_km(from_place, to_place)
        speed = {"driving": 45, "transit": 32, "walking": 5}.get(profile, 45)
        return {"from_place": from_place, "to_place": to_place, "found": True, "profile": profile,
                "distance_km": round(km, 1), "duration_min": int(round(km / speed * 60 + 15))}

    return mcp


# ---------------------------------------------------------------------------------- ferries
def build_ferries() -> FastMCP:
    mcp = FastMCP("sim-ferries")

    @mcp.tool
    def search_ferries(from_port: str, to_port: str, date: str) -> dict:
        """Search ferry sailings. Prices are per passenger in GBP."""
        route = FERRY_ROUTES.get((from_port, to_port)) or FERRY_ROUTES.get((to_port, from_port))
        if not route:
            return {"from_port": from_port, "to_port": to_port, "date": date, "sailings": []}
        base, dur = route
        rng = _rng("ferries", from_port, to_port, date)
        sailings = [{
            "id": f"F-{from_port[:3]}{to_port[:3]}-{date}-{i}", "operator": rng.choice(["Blue Star", "SeaJets", "Jadrolinija", "Caremar", "NLG"]),
            "from_port": from_port, "to_port": to_port, "date": date,
            "depart_time": f"{7 + 4 * i:02d}:{rng.choice(['00', '30'])}",
            "duration_min": int(dur * rng.uniform(0.7, 1.3)),
            "price": round(base * rng.uniform(0.8, 1.5), 2),
        } for i in range(3)]
        return {"from_port": from_port, "to_port": to_port, "date": date, "sailings": sailings}

    return mcp


# ---------------------------------------------------------------------------------- experiences
def build_experiences() -> FastMCP:
    mcp = FastMCP("sim-experiences")

    @mcp.tool
    def search_experiences(city: str, date: str, category: str | None = None) -> dict:
        """Search bookable experiences. Prices per person in GBP."""
        rng = _rng("experiences", city, date)
        items = []
        for i in range(8):
            cat = CATEGORIES[i % 4]
            items.append({"id": f"X-{city[:3].upper()}-{i}", "title": f"{city} {cat} tour {i + 1}",
                          "category": cat, "city": city, "date": date,
                          "rating": round(rng.uniform(3.8, 5.0), 1), "price": round(rng.uniform(22, 120), 2)})
        if category:
            items = [x for x in items if x["category"] == category]
        return {"city": city, "date": date, "experiences": items}

    return mcp


# ---------------------------------------------------------------------------------- weather
def build_weather() -> FastMCP:
    mcp = FastMCP("sim-weather")

    @mcp.tool
    def forecast(lat: float, lon: float, date: str) -> dict:
        """Daily forecast (climatological simulation)."""
        d = date
        dd = _dt.date.fromisoformat(d)
        rng = _rng("weather", round(lat, 2), round(lon, 2), d)
        warm = 30 - 0.45 * abs(lat) + 9 * math.sin((dd.month - 4) / 12 * 2 * math.pi)
        tmax = round(warm + rng.uniform(-3, 3), 1)
        precip = round(max(0.0, rng.gauss(2.0 if dd.month in (6, 7, 8) else 4.0, 3.0)), 1)
        return {"date": d, "lat": lat, "lon": lon, "temp_max_c": tmax, "temp_min_c": round(tmax - rng.uniform(6, 11), 1),
                "precip_mm": precip, "summary": "rain" if precip > 4 else ("sunny" if tmax > 22 else "mild")}

    return mcp


BUILDERS = {"flights": build_flights, "hotels": build_hotels, "geo": build_geo,
            "ferries": build_ferries, "experiences": build_experiences, "weather": build_weather}
