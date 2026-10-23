from symbolic_detective.ingestion.europeana_api import EuropeanaClient
from symbolic_detective.ingestion.met_api import harvest_met_artifacts

# Met Museum Department IDs for reference
MET_DEPARTMENTS = {
    1: "American Decorative Arts",
    3: "Ancient Near Eastern Art",
    4: "Arms and Armor",
    5: "Arts of Africa, Oceania, and the Americas",
    6: "Asian Art",
    7: "The Cloisters",
    8: "The Costume Institute",
    9: "Drawings and Prints",
    10: "Egyptian Art",
    11: "European Paintings",
    12: "European Sculpture and Decorative Arts",
    13: "Greek and Roman Art",
    14: "Islamic Art",
    15: "The Robert Lehman Collection",
    16: "The Libraries",
    17: "Medieval Art",
    18: "Musical Instruments",
    19: "Photographs",
    21: "Modern and Contemporary Art"
}

europeana_client = EuropeanaClient()

# =====================================================================
# 1. THE CORE TARGET: Ancient Near East (The Ishtar Context)
# =====================================================================
# Met API: Department 3 (Ancient Near Eastern Art)
# harvest_met_artifacts(query="star of Ishtar", department_id=3, limit=10)
# harvest_met_artifacts(query="eight-pointed star", department_id=3, limit=50)
# harvest_met_artifacts(query="cylinder seal star", department_id=3, limit=3)
# harvest_met_artifacts(query="boundary stone", limit=16)

# Europeana API: Mesopotamian Subjects
# europeana_client.ingest_collection("Ishtar star", max_items=25)
# europeana_client.ingest_collection("Kassite kudurru", max_items=5)
# europeana_client.ingest_collection("cylinder seal Ishtar", max_items=5)
# europeana_client.ingest_collection("Neo-Assyrian seal", max_items=5)

# =====================================================================
# 2. THE HOMOPLASY TARGET: Islamic Art (Independent Geometry)
# =====================================================================
# Met API: Department 14 (Islamic Art)
# harvest_met_artifacts(query="star", department_id=14, limit=50)
# harvest_met_artifacts(query="eight-pointed star", department_id=14, limit=3)
# harvest_met_artifacts(query="star tile", department_id=14, limit=3)

# =====================================================================
# 3. GLOBAL VARIANTS: 8-Pointed Stars in Navigation & Other Cultures
# =====================================================================
# Met API: Cross-departmental eight-pointed stars
# harvest_met_artifacts(query="eight-pointed star", limit=30)
# harvest_met_artifacts(query="eight-pointed star", department_id=4, limit=30)  # Arms and Armor
# harvest_met_artifacts(query="eight-pointed star", department_id=6, limit=30)  # Asian Art
# harvest_met_artifacts(query="eight-pointed star", department_id=12, limit=30) # European Sculpture/Decorative Arts

# Met API: Specific cultural and navigational variations
# harvest_met_artifacts(query="Compass", limit=25)
# harvest_met_artifacts(query="Compass Rose", limit=15)
# harvest_met_artifacts(query="Wind Rose", limit=5)
# harvest_met_artifacts(query="Rub el Hizb", limit=5)
# harvest_met_artifacts(query="Star of Lakshmi", limit=5)
# harvest_met_artifacts(query="Ashtalakshmi", limit=5)
# harvest_met_artifacts(query="LeMoyne Star", limit=5)
# harvest_met_artifacts(query="Lone Star", limit=5)

# Europeana API: Broad eight-pointed star search
# europeana_client.ingest_collection("eight-pointed star", max_items=35)

# =====================================================================
# 4. THE HARD NEGATIVE: Rosettes (Symbolic Allomorphy)
# =====================================================================
# Met API: Near Eastern Art (Assyrian reliefs, ivory carvings)
# harvest_met_artifacts(query="rosette", department_id=3, limit=40)
# harvest_met_artifacts(query="rosette plaque", department_id=3, limit=3)

# =====================================================================
# 5. THE HARD NEGATIVE: Five-Pointed Stars & Egyptian Astronomy
# =====================================================================
# Met API: Department 10 (Egyptian Art)
# harvest_met_artifacts(query="star", department_id=10, limit=30)
# harvest_met_artifacts(query="Egyptian astronomical ceiling", limit=15)
# harvest_met_artifacts(query="eye of horus", department_id=10, limit=12)
# harvest_met_artifacts(query="egyptian symbol", department_id=10, limit=12)
# harvest_met_artifacts(query="hieroglyphics", department_id=10, limit=30)
# harvest_met_artifacts(query="inscription", department_id=10, limit=5)
# harvest_met_artifacts(query="stela", department_id=10, limit=15)
# harvest_met_artifacts(query="cartouche", department_id=10, limit=5)

# Met API: Cross-departmental five-pointed stars
# harvest_met_artifacts(query="five-pointed star", department_id=1, limit=5)  # American Decorative Arts
# harvest_met_artifacts(query="five-pointed star", department_id=4, limit=5)  # Arms and Armor

# Europeana API: Five-pointed star search and Egyptian queries
# europeana_client.ingest_collection("five-pointed star", max_items=5)
# europeana_client.ingest_collection("Egyptian astronomical ceiling", max_items=15)
# europeana_client.ingest_collection("Pyramid of Unas ceiling", max_items=15)
# europeana_client.ingest_collection("Tomb of Senenmut star", max_items=15)

# =====================================================================
# 6. THE HARD NEGATIVE: Six-Pointed Stars (Hexagrams)
# =====================================================================
# Met API: Cross-departmental six-pointed stars
# harvest_met_artifacts(query="six-pointed star", department_id=4, limit=5)   # Arms and Armor
# harvest_met_artifacts(query="six-pointed star", department_id=6, limit=15)  # Asian Art
# harvest_met_artifacts(query="six-pointed star", department_id=14, limit=30) # Islamic Art
# harvest_met_artifacts(query="Hexagram", limit=15)
# harvest_met_artifacts(query="Magen", limit=15)

# harvest_met_artifacts(query="radial sunburst", department_id=1, limit=5)  # American Decorative Arts

# =====================================================================
# v0.2
# =====================================================================
# EIGHT-POINTED STAR
# harvest_met_artifacts(query="eight-pointed star", department_id=14, limit=40)
# harvest_met_artifacts(query="eight-pointed star", department_id=3, limit=30)
# harvest_met_artifacts(query="octagram", limit=25)
# harvest_met_artifacts(query="Star of Ishtar", limit=20)
#
# # SIX-POINTED STAR
# harvest_met_artifacts(query="six-pointed star",  limit=90)
# harvest_met_artifacts(query="six-pointed star", department_id=14, limit=90)
# harvest_met_artifacts(query="hexagram", limit=50)
# harvest_met_artifacts(query="Star of David", limit=40)
# harvest_met_artifacts(query="Seal of Solomon", department_id=14, limit=40)
#
# # FIVE-POINTED STAR
# harvest_met_artifacts(query="five-pointed star", department_id=10, limit=30)
# harvest_met_artifacts(query="five-pointed star", department_id=13, limit=30)
# harvest_met_artifacts(query="pentagram", limit=25)
#
# # EIGHT-PETAL ROSETTE
# harvest_met_artifacts(query="eight-petaled rosette", department_id=3, limit=30)
# harvest_met_artifacts(query="eight-petaled rosette", department_id=14, limit=30)
# harvest_met_artifacts(query="eight-petalled rosette", limit=25)
# harvest_met_artifacts(query="eight-petaled flower", limit=20)
#
# # GENERAL ROSETTE
# harvest_met_artifacts(query="rosette", department_id=3, limit=30)
# harvest_met_artifacts(query="rosette", department_id=13, limit=30)
# harvest_met_artifacts(query="floral rosette", department_id=14, limit=30)
#
# # RADIAL SUNBURST
# harvest_met_artifacts(query="sunburst", limit=80)
# harvest_met_artifacts(query="sunburst motif", limit=50)
# harvest_met_artifacts(query="radiating rays", limit=25)
# harvest_met_artifacts(query="solar motif", limit=25)
# harvest_met_artifacts(query="solar disk", limit=25)
# harvest_met_artifacts(query="sunray", limit=25)
# harvest_met_artifacts(query="monstrance", limit=25)
# harvest_met_artifacts(query="radiating disc", limit=25)

#
# # Maltese Crosse
# harvest_met_artifacts(query="Maltese Crosse", limit=20)
# harvest_met_artifacts(query="Cross Pattée", limit=20)

# Spiral
# harvest_met_artifacts(query="greec spiral", department_id=13, limit=15)
# harvest_met_artifacts(query="spiral", department_id=10, limit=15)
# harvest_met_artifacts(query="spiral", department_id=9, limit=15)
# harvest_met_artifacts(query="koru", limit=5)
# harvest_met_artifacts(query="maori", limit=15)
# harvest_met_artifacts(query="waka huia", limit=5)
# harvest_met_artifacts(query="maori", department_id=5, limit=5)

# Europeana Queries
# europeana_queries = {
    # "eight_pointed_star": [
    #     "eight-pointed star",
    #     "octagram",
    #     "eight pointed star",
    #     "étoile à huit branches",
    #     "estrella de ocho puntas",
    # ],
    # "six_pointed_star": [
    #     "six-pointed star",
    #     "hexagram",
    #     "Star of David",
    #     "Magen David",
    #     "étoile à six branches"
    # ],
    # "five_pointed_star": [
    #     "five-pointed star",
    #     "pentagram",
    #     "five pointed star",
    #     "étoile à cinq branches",
    # ],
    # "eight_petal_rosette": [
        # "eight-petaled rosette",
        # "eight-petalled rosette",
        # "eight-petal flower",
        # "roseta de ocho pétalos",
    # ],
    # "rosette": [
    #     "rosette",
    #     "floral rosette",
    #     "rosace",
    #     "roseta",
    # ],
    # "radial_sunburst": [
    #     "sunburst",
    #     "sunburst motif",
    #     "radiating sun",
    #     "rayonnant",
    #     "Sonnenstrahlen",
#     ],
# }
#
# for label, queries in europeana_queries.items():
#     for query in queries:
#         europeana_client.ingest_collection(query, max_items=30)
