"""Location resolution: offline gazetteer first, optional geocoding, never invented coordinates.

Status values
  verified   - matched a gazetteer entry (coordinates are approximate area centroids)
  geocoded   - returned by the optional OpenStreetMap Nominatim lookup (approximate, verify)
  manual     - entered by a human reviewer
  ambiguous  - a generic name that exists in many places; no coordinates are assigned
  unverified - a place-like phrase was extracted but could not be located
  unknown    - nothing extracted
"""
from __future__ import annotations

import csv
import json
import re
import time
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

import config

UNKNOWN = "Unknown"
# Generic names shared by many neighbourhoods. We refuse to guess coordinates for these.
AMBIGUOUS_NAMES = {
    "shivaji nagar", "gandhi nagar", "station road", "main road", "market road", "new colony",
    "ram nagar", "indira nagar", "nehru nagar", "ambedkar nagar", "hanuman nagar", "azad nagar",
}


def norm_name(name: str | None) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def is_unknown(name: str | None) -> bool:
    return norm_name(name) in {"", "unknown", "needs verification", "none", "n/a"}


class LocationResolver:
    def __init__(self, gazetteer_path: Path | str | None = None, enable_geocoding: bool | None = None):
        self.path = Path(gazetteer_path or config.GAZETTEER_CSV)
        self.enable_geocoding = config.ENABLE_GEOCODING if enable_geocoding is None else enable_geocoding
        self.entries: dict[str, dict] = {}
        self._cache_file = config.ARTIFACT_DIR / "geocode_cache.json"
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    entry = {"name": row["name"], "lat": float(row["lat"]), "lon": float(row["lon"]),
                             "region": row.get("region", "")}
                except (KeyError, ValueError):
                    continue
                for key in [row["name"], *[a for a in (row.get("aliases") or "").split("|") if a]]:
                    self.entries[norm_name(key)] = entry
        names = sorted(self.entries, key=len, reverse=True)
        self._pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.I) if names else None

    def find_in_text(self, text: str) -> list[tuple[str, int, int]]:
        """Gazetteer names mentioned in text as (canonical name, start, end)."""
        if not self._pattern:
            return []
        found, taken = [], []
        for m in self._pattern.finditer(text):
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((self.entries[norm_name(m.group(0))]["name"], m.start(), m.end()))
        return found

    def resolve(self, name: str | None) -> dict:
        key = norm_name(name)
        if is_unknown(name):
            return {"name": UNKNOWN, "status": "unknown", "lat": None, "lon": None, "source": None,
                    "note": "No location extracted. Needs verification."}
        if key in AMBIGUOUS_NAMES:
            return {"name": name, "status": "ambiguous", "lat": None, "lon": None, "source": None,
                    "note": "Generic place name that exists in many areas. Needs verification."}
        if key in self.entries:
            e = self.entries[key]
            return {"name": e["name"], "status": "verified", "lat": e["lat"], "lon": e["lon"],
                    "source": "gazetteer", "note": f"Approximate area centroid ({e['region']})."}
        if self.enable_geocoding:
            geo = self._geocode(name)
            if geo:
                return geo
        return {"name": name, "status": "unverified", "lat": None, "lon": None, "source": None,
                "note": "Place-like text found but could not be located. Needs verification."}

    # ---- optional online geocoding (OFF by default) -------------------------------------------------
    def _read_cache(self) -> dict:
        try:
            return json.loads(self._cache_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _geocode(self, name: str) -> dict | None:
        cache = self._read_cache()
        if norm_name(name) in cache:
            return cache[norm_name(name)] or None
        try:
            q = urllib.parse.urlencode({"q": name, "format": "json", "limit": 3})
            req = urllib.request.Request(f"https://nominatim.openstreetmap.org/search?{q}",
                                         headers={"User-Agent": config.GEOCODER_USER_AGENT})
            time.sleep(1.1)  # Nominatim usage policy: max 1 request/second
            with urllib.request.urlopen(req, timeout=6) as resp:
                hits = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None  # offline or blocked: stay "unverified", never guess
        if not hits:
            result = None
        else:
            lats = [float(h["lat"]) for h in hits]
            lons = [float(h["lon"]) for h in hits]
            spread = max(max(lats) - min(lats), max(lons) - min(lons))
            if spread > 0.5:  # several far-apart matches -> ambiguous, no coordinates
                result = {"name": name, "status": "ambiguous", "lat": None, "lon": None, "source": "nominatim",
                          "note": f"{len(hits)} different places match. Needs verification."}
            else:
                result = {"name": name, "status": "geocoded", "lat": lats[0], "lon": lons[0],
                          "source": "nominatim", "note": "Approximate result from OpenStreetMap. Verify."}
        try:
            cache[norm_name(name)] = result
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._cache_file.write_text(json.dumps(cache), encoding="utf-8")
        except Exception:
            pass
        return result


@lru_cache(maxsize=1)
def get_resolver() -> LocationResolver:
    return LocationResolver()
