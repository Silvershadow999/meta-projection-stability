import time
import requests
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class GlobalSenseConfig:
    """
    Konfiguration für GlobalSense.
    cache_ttl_seconds:
        Wie lange API-Werte im Cache bleiben sollen.
    world_code:
        'WLD' = Weltaggregat (empfohlen für globale Trigger).
    """
    cache_ttl_seconds: int = 60 * 60 * 6  # 6 Stunden
    world_code: str = "WLD"


class GlobalSense:
    """
    Holt reale Welt-Daten (World Bank API) und mappt sie auf Simulationsparameter.

    Features:
    - Robuster API-Zugriff (Timeout, Statusprüfung, JSON-Checks)
    - Fallback-Werte
    - Caching
    - Mehrere Indikatoren + gewichteter global_stress_index
    """

    BASE_URL = "https://api.worldbank.org/v2"

    # ---- Indikator-Katalog (World Bank Codes) ----
    # Du kannst diese Liste später erweitern.
    INDICATORS = {
        "gdp_growth": "NY.GDP.MKTP.KD.ZG",      # GDP growth (annual %)
        "co2_per_capita": "EN.ATM.CO2E.PC",     # CO2 emissions (metric tons per capita)
        "inflation_cpi": "FP.CPI.TOTL.ZG",      # Inflation, consumer prices (annual %)
        "unemployment": "SL.UEM.TOTL.ZS",       # Unemployment, total (% of total labor force)
    }

    def __init__(self, gs_config: Optional[GlobalSenseConfig] = None):
        self.gs_config = gs_config or GlobalSenseConfig()

        # Cache-Struktur:
        # { (country, indicator_code): (timestamp, value_float) }
        self._cache: Dict[Tuple[str, str], Tuple[float, float]] = {}

    # ---------------------------------------------------------------------
    # 1) Rohwert laden (ein Indikator)
    # ---------------------------------------------------------------------
    def get_global_metric(
        self,
        indicator: str = "NY.GDP.MKTP.KD.ZG",
        country: Optional[str] = None,
        fallback: float = 1.0,
        force_refresh: bool = False,
        per_page: int = 12
    ) -> float:
        """
        Holt den neuesten verfügbaren (nicht-null) Wert eines World-Bank-Indikators.

        Args:
            indicator: World Bank indicator code (z. B. NY.GDP.MKTP.KD.ZG)
            country: standardmäßig Weltaggregat (WLD)
            fallback: Rückgabewert bei Fehlern
            force_refresh: ignoriert Cache
            per_page: wie viele Einträge abgefragt werden (für None-Filter)

        Returns:
            float (neuester verwertbarer Wert)
        """
        country = country or self.gs_config.world_code
        cache_key = (country, indicator)

        # ---- Cache prüfen ----
        if not force_refresh and cache_key in self._cache:
            ts, cached_value = self._cache[cache_key]
            age = time.time() - ts
            if age < self.gs_config.cache_ttl_seconds:
                return cached_value

        # ---- Korrekte World-Bank-URL ----
        # Beispiel: /v2/country/WLD/indicator/NY.GDP.MKTP.KD.ZG
        url = f"{self.BASE_URL}/country/{country}/indicator/{indicator}"
        params = {
            "format": "json",
            "per_page": per_page,
            "mrnev": per_page  # most recent N values
        }

        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()

            payload = resp.json()

            # Erwartete Struktur: [metadata, [rows...]]
            if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
                raise ValueError(f"Unerwartetes API-Format für Indikator {indicator}")

            # Ersten nicht-None-Wert finden
            for row in payload[1]:
                value = row.get("value")
                if value is not None:
                    value_f = float(value)
                    self._cache[cache_key] = (time.time(), value_f)
                    return value_f

            print(f"[GlobalSense] Kein verwertbarer Wert für {indicator}. Nutze Fallback={fallback}")
            return fallback

        except requests.RequestException as e:
            print(f"[GlobalSense] Netzwerk/API-Fehler für {indicator}: {e}. Nutze Fallback={fallback}")
            return fallback
        except (ValueError, TypeError, KeyError) as e:
            print(f"[GlobalSense] Antwortfehler für {indicator}: {e}. Nutze Fallback={fallback}")
            return fallback

    # ---------------------------------------------------------------------
    # 2) Mehrere Indikatoren sammeln
    # ---------------------------------------------------------------------
    def collect_global_snapshot(self, force_refresh: bool = False) -> Dict[str, float]:
        """
        Holt mehrere globale Metriken (Weltaggregat) in einem Snapshot.
        """
        snapshot = {
            "gdp_growth": self.get_global_metric(
                indicator=self.INDICATORS["gdp_growth"],
                fallback=0.0,
                force_refresh=force_refresh
            ),
            "co2_per_capita": self.get_global_metric(
                indicator=self.INDICATORS["co2_per_capita"],
                fallback=4.5,   # grober neutraler Fallback
                force_refresh=force_refresh
            ),
            "inflation_cpi": self.get_global_metric(
                indicator=self.INDICATORS["inflation_cpi"],
                fallback=3.0,
                force_refresh=force_refresh
            ),
            "unemployment": self.get_global_metric(
                indicator=self.INDICATORS["unemployment"],
                fallback=6.0,
                force_refresh=force_refresh
            ),
        }
        return snapshot

    # ---------------------------------------------------------------------
    # 3) Normalisierung / Skalierung
    # ---------------------------------------------------------------------
    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @classmethod
    def _normalize_metric(cls, name: str, value: float) -> float:
        """
        Normalisiert Metriken auf ungefähr [0..1] (0 = entspannt / 1 = stressig)
        Heuristische Grenzen (anpassbar je nach Modellkalibrierung).
        """
        if name == "gdp_growth":
            # Wachstum hoch = weniger Stress; Rezession = mehr Stress
            # Mapping: -5% -> 1.0 Stress, +5% -> 0.0 Stress
            return cls._clamp((5.0 - value) / 10.0, 0.0, 1.0)

        elif name == "inflation_cpi":
            # z. B. 0..10% auf 0..1 (höher = stressiger)
            return cls._clamp(value / 10.0, 0.0, 1.0)

        elif name == "unemployment":
            # 0..15% auf 0..1
            return cls._clamp(value / 15.0, 0.0, 1.0)

        elif name == "co2_per_capita":
            # Kein "Stress" im kurzfristig-ökonomischen Sinn, aber globaler Systemdruck.
            # Heuristik 0..20 t/Kopf -> 0..1
            return cls._clamp(value / 20.0, 0.0, 1.0)

        # Unbekannt: neutral
        return 0.5

    # ---------------------------------------------------------------------
    # 4) Gewichteter Score
    # ---------------------------------------------------------------------
    def compute_global_stress_index(
        self,
        snapshot: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
        force_refresh: bool = False
    ) -> float:
        """
        Berechnet einen gewichteten global_stress_index in [0..1].

        0.0 = entspannt / günstig
        1.0 = hochstressige globale Lage
        """
        if snapshot is None:
            snapshot = self.collect_global_snapshot(force_refresh=force_refresh)

        # Standardgewichte (Summe muss nicht exakt 1 sein; wird normiert)
        if weights is None:
            weights = {
                "gdp_growth": 0.40,
                "inflation_cpi": 0.25,
                "unemployment": 0.20,
                "co2_per_capita": 0.15,
            }

        weighted_sum = 0.0
        weight_total = 0.0

        for key, raw_val in snapshot.items():
            w = float(weights.get(key, 0.0))
            if w <= 0:
                continue
            norm_val = self._normalize_metric(key, raw_val)
            weighted_sum += w * norm_val
            weight_total += w

        if weight_total <= 0:
            return 0.5  # neutral fallback

        return self._clamp(weighted_sum / weight_total, 0.0, 1.0)

    # ---------------------------------------------------------------------
    # 5) Mapping auf Simulations-Config (sanfte Anpassung)
    # ---------------------------------------------------------------------
    def map_to_config(
        self,
        config,
        use_stress_index: bool = True,
        force_refresh: bool = False,
        debug: bool = False
    ):
        """
        Passt die Simulation an reale Makrodynamik an.

        Erwartet mindestens:
            config.human_recovery_base (float)

        Optional (wenn vorhanden):
            config.noise_scale
            config.system_drag
            config.adaptation_rate
        """
        if not hasattr(config, "human_recovery_base"):
            raise AttributeError("config benötigt das Attribut 'human_recovery_base'")

        # Basiswerte merken, damit mehrfaches Aufrufen nicht driftet
        if not hasattr(config, "_baseline_human_recovery_base"):
            config._baseline_human_recovery_base = float(config.human_recovery_base)

        if use_stress_index:
            snapshot = self.collect_global_snapshot(force_refresh=force_refresh)
            stress = self.compute_global_stress_index(snapshot=snapshot)

            # Sanftes Mapping:
            # stress=0.0 -> factor ~1.15
            # stress=0.5 -> factor ~1.00
            # stress=1.0 -> factor ~0.75
            recovery_factor = self._clamp(1.15 - 0.40 * stress, 0.75, 1.15)

            config.human_recovery_base = config._baseline_human_recovery_base * recovery_factor

            # Optional weitere Parameter anpassen (nur wenn vorhanden)
            if hasattr(config, "noise_scale"):
                if not hasattr(config, "_baseline_noise_scale"):
                    config._baseline_noise_scale = float(config.noise_scale)
                # Mehr globaler Stress -> etwas mehr Störrauschen
                noise_factor = self._clamp(0.95 + 0.25 * stress, 0.8, 1.3)
                config.noise_scale = config._baseline_noise_scale * noise_factor

            if hasattr(config, "system_drag"):
                if not hasattr(config, "_baseline_system_drag"):
                    config._baseline_system_drag = float(config.system_drag)
                # Mehr Stress -> mehr "Drag"
                drag_factor = self._clamp(0.95 + 0.30 * stress, 0.8, 1.4)
                config.system_drag = config._baseline_system_drag * drag_factor

            if hasattr(config, "adaptation_rate"):
                if not hasattr(config, "_baseline_adaptation_rate"):
                    config._baseline_adaptation_rate = float(config.adaptation_rate)
                # Mehr Stress -> etwas geringere Adaptionsrate
                adapt_factor = self._clamp(1.05 - 0.25 * stress, 0.75, 1.1)
                config.adaptation_rate = config._baseline_adaptation_rate * adapt_factor

            if debug:
                print("[GlobalSense] Snapshot:", snapshot)
                print(f"[GlobalSense] stress_index={stress:.3f}, recovery_factor={recovery_factor:.3f}")

            return config

        else:
            # Minimalmodus: nur GDP-Wachstum (dein ursprünglicher Ansatz, aber korrekt + weich)
            growth = self.get_global_metric(
                indicator=self.INDICATORS["gdp_growth"],
                fallback=0.0,
                force_refresh=force_refresh
            )

            if not hasattr(config, "_baseline_human_recovery_base"):
                config._baseline_human_recovery_base = float(config.human_recovery_base)

            # Weiches Mapping auf Basis GDP growth
            # Beispiel: factor = 1 + 0.03 * growth, begrenzt
            recovery_factor = self._clamp(1.0 + 0.03 * growth, 0.75, 1.20)
            config.human_recovery_base = config._baseline_human_recovery_base * recovery_factor

            if debug:
                print(f"[GlobalSense] gdp_growth={growth:.3f}, recovery_factor={recovery_factor:.3f}")

            return config

    # ---------------------------------------------------------------------
    # 6) Cache-Management (optional)
    # ---------------------------------------------------------------------
    def clear_cache(self):
        self._cache.clear()

    def cache_info(self) -> Dict[str, int]:
        return {
            "entries": len(self._cache),
            "ttl_seconds": self.gs_config.cache_ttl_seconds
        }


# -------------------------------------------------------------------------
# Beispiel-Verwendung
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Beispielhafte Simulations-Config
    class SimConfig:
        def __init__(self):
            self.human_recovery_base = 1.0
            self.noise_scale = 0.2
            self.system_drag = 0.1
            self.adaptation_rate = 0.05

        def __repr__(self):
            return (
                f"SimConfig("
                f"human_recovery_base={self.human_recovery_base:.4f}, "
                f"noise_scale={self.noise_scale:.4f}, "
                f"system_drag={self.system_drag:.4f}, "
                f"adaptation_rate={self.adaptation_rate:.4f})"
            )

    cfg = SimConfig()
    gs = GlobalSense()

    print("Vorher:", cfg)
    cfg = gs.map_to_config(cfg, use_stress_index=True, debug=True)
    print("Nachher:", cfg)
    print("Cache:", gs.cache_info())
