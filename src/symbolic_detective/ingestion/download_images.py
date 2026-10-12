from symbolic_detective.ingestion.europeana_api import EuropeanaClient
from symbolic_detective.ingestion.met_api import harvest_met_artifacts

# 1. The Core Target: Ancient Near East (The Ishtar Context)
# Ancient Near Eastern Art (Department 3)
# harvest_met_artifacts(query="star of Ishtar", department_id=3, limit=10)
# Search for literal star mentions in Near Eastern Art
# harvest_met_artifacts(query="eight-pointed star", department_id=3, limit=50)
# Search for cylinder seals
# harvest_met_artifacts(query="cylinder seal star", department_id=3, limit=3)

# harvest_met_artifacts(query="Compass Rose", limit=15)
# harvest_met_artifacts(query="Wind Rose", limit=5)
# harvest_met_artifacts(query="Rub el Hizb", limit=5)
# harvest_met_artifacts(query="LeMoyne Star", limit=5)
# harvest_met_artifacts(query="Lone Star", limit=5)
# harvest_met_artifacts(query="Star of Lakshmi", limit=5)
# harvest_met_artifacts(query="Ashtalakshmi", limit=5)
# harvest_met_artifacts(query="Compass", limit=25)

# harvest_met_artifacts(query="boundary stone", limit=16)

# 2. The Homoplasy Target: Islamic Art (Independent Geometry)
# Islamic Art (Department 14)
# Harvest eight-pointed star tiles and ceramics
# harvest_met_artifacts(query="eight-pointed star", department_id=14, limit=3)
# Harvest star patterns in woodwork and metalwork
# harvest_met_artifacts(query="star tile", department_id=14, limit=3)

# 3. The Hard Negative: Rosettes (Symbolic Allomorphy)
# Harvest rosettes from Near Eastern Art (Assyrian reliefs, ivory carvings)
# harvest_met_artifacts(query="rosette", department_id=3, limit=40)
# Specific search for ivory rosettes (very common in Nimrud)
# harvest_met_artifacts(query="rosette plaque", department_id=3, limit=3)

# 4. The Hard Negative: Five-Pointed Stars
# Egyptian Art (Department 10)
# harvest_met_artifacts(query="star", department_id=10, limit=30)
# harvest_met_artifacts(query="eye of horus", department_id=10, limit=12)
# harvest_met_artifacts(query="egyptian symbol", department_id=10, limit=12)
# harvest_met_artifacts(query="hieroglyphics", department_id=10, limit=30)
# harvest_met_artifacts(query="inscription", department_id=10, limit=5)
# harvest_met_artifacts(query="stela", department_id=10, limit=15)
# harvest_met_artifacts(query="cartouche", department_id=10, limit=5)

# harvest_met_artifacts(query="five-pointed star", department_id=1, limit=5)
# harvest_met_artifacts(query="five-pointed star", department_id=4, limit=5)
# harvest_met_artifacts(query="six-pointed star", department_id=4, limit=5)
# harvest_met_artifacts(query="six-pointed star", department_id=6, limit=15)
# harvest_met_artifacts(query="six-pointed star", department_id=14, limit=30)
# harvest_met_artifacts(query="Hexagram", limit=15)
# harvest_met_artifacts(query="Magen", limit=15)
# harvest_met_artifacts(query="star", department_id=14, limit=50)

# EUROPEANA API
client = EuropeanaClient()

client.ingest_collection("Ishtar star", max_items=25)
client.ingest_collection("eight-pointed star", max_items=35)
# client.ingest_collection("five-pointed star", max_items=5)

# Egypt
"Egyptian astronomical ceiling"
"Pyramid of Unas ceiling"
"Tomb of Senenmut star"
# harvest_met_artifacts(query="Egyptian astronomical ceiling", limit=15)

#  The Mesopotamian Star of Ishtar
# client.ingest_collection("Kassite kudurru", 5)
# client.ingest_collection("cylinder seal Ishtar", 5)
# client.ingest_collection("Neo-Assyrian seal", 5)

# harvest_met_artifacts(query="eight-pointed star", limit=30)
# harvest_met_artifacts(query="eight-pointed star", department_id=4, limit=30)
# harvest_met_artifacts(query="eight-pointed star", department_id=12, limit=30)
# harvest_met_artifacts(query="eight-pointed star", department_id=6, limit=30)
