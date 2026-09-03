"""Kanal-Woerterbuch der 55 Cache-Kanaele.

WICHTIGE UNTERSCHEIDUNG - zwei Ebenen mit unterschiedlicher Sicherheit:

1. **Kanal -> (Modalitaet, Bandnummer)**: EMPIRISCH VERIFIZIERT in Stufe 5
   durch exakten uint8-Abgleich der 61-Schritt-Reihen gegen die Rohraster.
   Diese Zuordnung ist belegt, nicht angenommen.

2. **Bandnummer -> physikalische Variable**: UEBERNOMMEN aus der Tabelle im
   README des Datensatzes. Die GeoTIFFs tragen nur generische Bandnamen
   ("Band_1" ...) und keine Einheiten, daher ist die Reihenfolge innerhalb
   einer Modalitaet NICHT aus den Dateien verifizierbar. Stufe 2b liefert
   dazu einen Plausibilitaetstest (Sommer-/Winterkontrast), der grobe
   Vertauschungen sichtbar macht, aber keinen Beweis darstellt.

Alle Raster sind uint8 ohne scale/offset. Die physikalischen Einheiten der
Originalprodukte sind daher NICHT rekonstruierbar; die Spalte "Einheit"
nennt die Einheit der QUELLE, nicht die der gespeicherten Werte.
"""

from __future__ import annotations

# (Modalitaet, Band, Kurzname, Beschreibung, Einheit der Quelle, Kadenz, Rolle)
CHANNELS: list[dict] = [
    # ---------------------------------------------------------------- Label
    dict(ch=0, mod="MCD14A1", band=0, name="active_fire",
         desc="Aktives Feuer, binaer 0/1, aus MODIS Thermal Anomalies",
         unit="binaer", cadence="taeglich", role="LABEL",
         note="Im Cache als EINGABEKANAL 0 vorhanden - Hauptquelle des Leakage"),

    # ---------------------------------------------------------------- ERA5-Land
    dict(ch=1, mod="ERA5", band=0, name="temperature_2m",
         desc="Lufttemperatur 2 m ueber Grund, Tagesmittel", unit="K",
         cadence="taeglich", role="Wetter"),
    dict(ch=2, mod="ERA5", band=1, name="u_component_of_wind_10m",
         desc="Windkomponente Ost-West in 10 m", unit="m/s",
         cadence="taeglich", role="Wetter"),
    dict(ch=3, mod="ERA5", band=2, name="v_component_of_wind_10m",
         desc="Windkomponente Nord-Sued in 10 m", unit="m/s",
         cadence="taeglich", role="Wetter"),
    dict(ch=4, mod="ERA5", band=3, name="snow_cover",
         desc="Schneebedeckungsgrad der Gitterzelle", unit="%",
         cadence="taeglich", role="Wetter"),
    dict(ch=5, mod="ERA5", band=4, name="total_precipitation_sum",
         desc="Niederschlagssumme des Tages", unit="m",
         cadence="taeglich", role="Wetter"),
    dict(ch=6, mod="ERA5", band=5, name="surface_latent_heat_flux_sum",
         desc="Latenter Waermestrom an der Oberflaeche, Tagessumme", unit="J/m2",
         cadence="taeglich", role="Wetter"),
    dict(ch=7, mod="ERA5", band=6, name="dewpoint_temperature_2m",
         desc="Taupunkttemperatur 2 m - zusammen mit T2m die Luftfeuchte",
         unit="K", cadence="taeglich", role="Wetter"),
    dict(ch=8, mod="ERA5", band=7, name="surface_pressure",
         desc="Luftdruck an der Oberflaeche", unit="Pa",
         cadence="taeglich", role="Wetter"),
    dict(ch=9, mod="ERA5", band=8, name="volumetric_soil_water_layer_1",
         desc="Bodenwassergehalt 0-7 cm", unit="m3/m3",
         cadence="taeglich", role="Bodenfeuchte"),
    dict(ch=10, mod="ERA5", band=9, name="volumetric_soil_water_layer_2",
         desc="Bodenwassergehalt 7-28 cm", unit="m3/m3",
         cadence="taeglich", role="Bodenfeuchte"),
    dict(ch=11, mod="ERA5", band=10, name="volumetric_soil_water_layer_3",
         desc="Bodenwassergehalt 28-100 cm", unit="m3/m3",
         cadence="taeglich", role="Bodenfeuchte"),
    dict(ch=12, mod="ERA5", band=11, name="volumetric_soil_water_layer_4",
         desc="Bodenwassergehalt 100-289 cm", unit="m3/m3",
         cadence="taeglich", role="Bodenfeuchte"),
    dict(ch=13, mod="ERA5", band=12, name="temperature_2m_max",
         desc="Tagesmaximum der Lufttemperatur 2 m", unit="K",
         cadence="taeglich", role="Wetter"),
    dict(ch=14, mod="ERA5", band=13, name="skin_temperature_max",
         desc="Tagesmaximum der Oberflaechentemperatur", unit="K",
         cadence="taeglich", role="Wetter"),
    dict(ch=15, mod="ERA5", band=14, name="potential_evaporation_sum",
         desc="Potenzielle Verdunstung, Tagessumme", unit="m",
         cadence="taeglich", role="Trockenheit"),
    dict(ch=16, mod="ERA5", band=15, name="total_evaporation_sum",
         desc="Tatsaechliche Verdunstung, Tagessumme", unit="m",
         cadence="taeglich", role="Trockenheit"),
    dict(ch=17, mod="ERA5", band=16, name="skin_reservoir_content",
         desc="Wasser auf Vegetation und Boden (Interzeption)", unit="m",
         cadence="taeglich", role="Bodenfeuchte"),
    dict(ch=18, mod="ERA5", band=17, name="surface_net_solar_radiation_sum",
         desc="Netto-Sonneneinstrahlung an der Oberflaeche, Tagessumme",
         unit="J/m2", cadence="taeglich", role="Strahlung"),

    # ---------------------------------------------------------------- CFFDRS / FWI
    dict(ch=19, mod="FWI", band=0, name="FFMC",
         desc="Fine Fuel Moisture Code - Feuchte der feinen Streu, "
              "reagiert innerhalb von Stunden; steuert Entzuendbarkeit",
         unit="Index 0-101", cadence="taeglich", role="Feuerwetter"),
    dict(ch=20, mod="FWI", band=1, name="DMC",
         desc="Duff Moisture Code - Feuchte der lockeren Humusschicht, "
              "Zeitkonstante Tage bis Wochen",
         unit="Index", cadence="taeglich", role="Feuerwetter"),
    dict(ch=21, mod="FWI", band=2, name="DC",
         desc="Drought Code - Feuchte tiefer, kompakter Organik, "
              "Zeitkonstante Wochen bis Monate; Saisondurre",
         unit="Index", cadence="taeglich", role="Feuerwetter"),
    dict(ch=22, mod="FWI", band=3, name="ISI",
         desc="Initial Spread Index - erwartete Ausbreitungsgeschwindigkeit "
              "aus FFMC und Wind",
         unit="Index", cadence="taeglich", role="Feuerwetter"),
    dict(ch=23, mod="FWI", band=4, name="BUI",
         desc="Buildup Index - verfuegbare Brennstoffmenge aus DMC und DC",
         unit="Index", cadence="taeglich", role="Feuerwetter"),
    dict(ch=24, mod="FWI", band=5, name="FWI",
         desc="Fire Weather Index - Gesamtindex aus ISI und BUI; "
              "die operationelle Waldbrandgefahr Kanadas",
         unit="Index", cadence="taeglich", role="Feuerwetter"),

    # ---------------------------------------------------------------- VPD
    dict(ch=25, mod="VPD", band=0, name="vapor_pressure_deficit",
         desc="Dampfdruckdefizit - wie stark die Luft der Vegetation Wasser "
              "entzieht; einer der staerksten Einzelpraediktoren fuer Brand",
         unit="kPa", cadence="taeglich", role="Trockenheit"),

    # ---------------------------------------------------------------- OSM
    dict(ch=26, mod="OSMs", band=0, name="road_density",
         desc="Strassendichte - Zugaenglichkeit und menschliche Zuendquellen",
         unit="Dichte", cadence="statisch", role="Anthropogen"),
    dict(ch=27, mod="OSMs", band=1, name="powerline_density",
         desc="Leitungsdichte - technische Zuendquelle", unit="Dichte",
         cadence="statisch", role="Anthropogen"),
    dict(ch=28, mod="OSMs", band=2, name="building_density",
         desc="Gebaeudedichte - Siedlungsnaehe, WUI-Indikator", unit="Dichte",
         cadence="statisch", role="Anthropogen"),
    dict(ch=29, mod="OSMs", band=3, name="water_density",
         desc="Gewaesserdichte - natuerliche Ausbreitungsbarriere",
         unit="Dichte", cadence="statisch", role="Anthropogen"),

    # ---------------------------------------------------------------- WorldPop
    dict(ch=30, mod="Worldpop", band=0, name="population_density",
         desc="Bevoelkerungsdichte", unit="Einwohner/Zelle",
         cadence="jaehrlich", role="Anthropogen"),

    # ---------------------------------------------------------------- Terrain
    dict(ch=31, mod="DEMs", band=0, name="DEM",
         desc="Gelaendehoehe ueber Meeresniveau", unit="m",
         cadence="statisch", role="Terrain"),
    dict(ch=32, mod="DEMs", band=1, name="slope",
         desc="Hangneigung - Feuer laeuft hangaufwaerts schneller", unit="Grad",
         cadence="statisch", role="Terrain"),
    dict(ch=33, mod="DEMs", band=2, name="aspect_sin",
         desc="Exposition, Sinuskomponente (Ost-West-Anteil)", unit="-1..1",
         cadence="statisch", role="Terrain"),
    dict(ch=34, mod="DEMs", band=3, name="aspect_cos",
         desc="Exposition, Kosinuskomponente (Nord-Sued-Anteil); "
              "Suedhaenge trocknen staerker aus", unit="-1..1",
         cadence="statisch", role="Terrain"),
    dict(ch=35, mod="DEMs", band=4, name="hillshade",
         desc="Schummerung - Proxy fuer Einstrahlungsgeometrie", unit="0-255",
         cadence="statisch", role="Terrain"),
    dict(ch=36, mod="DEMs", band=5, name="TPI",
         desc="Topographic Position Index - Kuppe, Hang oder Senke",
         unit="Index", cadence="statisch", role="Terrain"),
    dict(ch=37, mod="DEMs", band=6, name="TWI",
         desc="Topographic Wetness Index - wo sich Wasser sammelt",
         unit="Index", cadence="statisch", role="Terrain"),

    # ---------------------------------------------------------------- Vegetation
    dict(ch=38, mod="NDVI_EVI", band=0, name="NDVI",
         desc="Normalized Difference Vegetation Index - Vegetationsmenge "
              "und -vitalitaet; Proxy fuer Brennstoff", unit="-1..1",
         cadence="taeglich", role="Vegetation"),
    dict(ch=39, mod="NDVI_EVI", band=1, name="EVI",
         desc="Enhanced Vegetation Index - wie NDVI, robuster in dichter "
              "Vegetation", unit="-1..1", cadence="taeglich", role="Vegetation"),
    dict(ch=40, mod="MCD15A3H", band=0, name="LAI",
         desc="Leaf Area Index - Blattflaeche je Bodenflaeche, Biomasseproxy",
         unit="m2/m2", cadence="taeglich (aus 4-Tage-Komposit interpoliert)",
         role="Vegetation",
         note="Interpolation kann Information ueber den Zuendzeitpunkt hinweg tragen"),
    dict(ch=41, mod="MCD15A3H", band=1, name="FPAR",
         desc="Anteil absorbierter photosynthetisch aktiver Strahlung",
         unit="0-1", cadence="taeglich (aus 4-Tage-Komposit interpoliert)",
         role="Vegetation",
         note="Interpolation kann Information ueber den Zuendzeitpunkt hinweg tragen"),

    # ---------------------------------------------------------------- Reflektanz
    dict(ch=42, mod="MCD09GA", band=0, name="sur_refl_b01",
         desc="Oberflaechenreflexion Band 1 (rot, 620-670 nm)",
         unit="Reflexion", cadence="taeglich", role="Reflektanz",
         note="QA-gefiltert und gapfilled"),
    dict(ch=43, mod="MCD09GA", band=1, name="sur_refl_b02",
         desc="Oberflaechenreflexion Band 2 (nahes Infrarot, 841-876 nm)",
         unit="Reflexion", cadence="taeglich", role="Reflektanz",
         note="QA-gefiltert und gapfilled"),
    dict(ch=44, mod="MCD09GA", band=2, name="sur_refl_b03",
         desc="Oberflaechenreflexion Band 3 (blau, 459-479 nm)",
         unit="Reflexion", cadence="taeglich", role="Reflektanz",
         note="QA-gefiltert und gapfilled"),
    dict(ch=45, mod="MCD09GA", band=3, name="sur_refl_b07",
         desc="Oberflaechenreflexion Band 7 (kurzwelliges IR, 2105-2155 nm); "
              "reagiert auf Brandnarben und Blattwassergehalt",
         unit="Reflexion", cadence="taeglich", role="Reflektanz",
         note="QA-gefiltert und gapfilled"),

    # ---------------------------------------------------------------- LST
    dict(ch=46, mod="MCD11A1", band=0, name="LST_Day_1km",
         desc="Landoberflaechentemperatur tagsueber", unit="K",
         cadence="taeglich", role="Oberflaechentemperatur",
         note="QA-gefiltert und gapfilled"),
    dict(ch=47, mod="MCD11A1", band=1, name="LST_Night_1km",
         desc="Landoberflaechentemperatur nachts", unit="K",
         cadence="taeglich", role="Oberflaechentemperatur",
         note="QA-gefiltert und gapfilled"),
    dict(ch=48, mod="MCD11A1", band=2, name="Emis_31",
         desc="Emissivitaet Band 31", unit="0-1", cadence="taeglich",
         role="Oberflaechentemperatur"),
    dict(ch=49, mod="MCD11A1", band=3, name="Emis_32",
         desc="Emissivitaet Band 32", unit="0-1", cadence="taeglich",
         role="Oberflaechentemperatur"),

    # ---------------------------------------------------------------- Helligkeitstemperatur
    dict(ch=50, mod="MCD09CMG", band=0, name="BT_band20",
         desc="Helligkeitstemperatur Band 20 (3.66-3.84 um) - das Fenster, "
              "in dem MODIS aktive Feuer detektiert",
         unit="K", cadence="taeglich", role="Helligkeitstemperatur",
         note="LEAKAGE-VERDACHT: dieselbe physikalische Groesse, aus der das "
              "Label MCD14A1 abgeleitet wird"),
    dict(ch=51, mod="MCD09CMG", band=1, name="BT_band21",
         desc="Helligkeitstemperatur Band 21 (3.93-3.99 um) - Feuerkanal "
              "mit hoher Saettigungsgrenze",
         unit="K", cadence="taeglich", role="Helligkeitstemperatur",
         note="LEAKAGE-VERDACHT: Feuerdetektionskanal von MODIS"),
    dict(ch=52, mod="MCD09CMG", band=2, name="BT_band31",
         desc="Helligkeitstemperatur Band 31 (10.78-11.28 um) - "
              "Referenzkanal der Feuerdetektion",
         unit="K", cadence="taeglich", role="Helligkeitstemperatur",
         note="LEAKAGE-VERDACHT: geht in den MODIS-Feueralgorithmus ein"),
    dict(ch=53, mod="MCD09CMG", band=3, name="BT_band32",
         desc="Helligkeitstemperatur Band 32 (11.77-12.27 um)",
         unit="K", cadence="taeglich", role="Helligkeitstemperatur",
         note="LEAKAGE-VERDACHT: geht in den MODIS-Feueralgorithmus ein"),

    # ---------------------------------------------------------------- Landbedeckung
    dict(ch=54, mod="MCD12Q1", band=0, name="landcover_IGBP",
         desc="Landbedeckungsklasse nach IGBP (1-16 Land, 17 Wasser, "
              "0 unklassifiziert) - KATEGORIAL, nicht z-normalisieren",
         unit="Klasse", cadence="jaehrlich", role="Statisch/kategorial"),
]

BY_INDEX = {c["ch"]: c for c in CHANNELS}
BY_MOD_BAND = {(c["mod"], c["band"]): c for c in CHANNELS}

IGBP_CLASSES = {
    0: "unklassifiziert / Fuellwert",
    1: "Immergruener Nadelwald",
    2: "Immergruener Laubwald",
    3: "Laubabwerfender Nadelwald",
    4: "Laubabwerfender Laubwald",
    5: "Mischwald",
    6: "Geschlossenes Buschland",
    7: "Offenes Buschland",
    8: "Waldsavanne",
    9: "Savanne",
    10: "Grasland",
    11: "Dauerfeuchtgebiet",
    12: "Ackerland",
    13: "Stadt und bebaut",
    14: "Ackerland-Naturvegetation-Mosaik",
    15: "Schnee und Eis",
    16: "Vegetationsarm",
    17: "Wasserflaechen",
}
