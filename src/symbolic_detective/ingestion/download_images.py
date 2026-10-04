from symbolic_detective.ingestion.met_api import harvest_met_artifacts

# 1. The Core Target: Ancient Near East (The Ishtar Context)
# Ancient Near Eastern Art (Department 3)
# harvest_met_artifacts(query="star of Ishtar", department_id=3, limit=10)
# Search for literal star mentions in Near Eastern Art
harvest_met_artifacts(query="eight-pointed star", department_id=3, limit=10)
# Search for cylinder seals
# harvest_met_artifacts(query="cylinder seal star", department_id=3, limit=3)

# 2. The Homoplasy Target: Islamic Art (Independent Geometry)
# Islamic Art (Department 14)
# Harvest eight-pointed star tiles and ceramics
# harvest_met_artifacts(query="eight-pointed star", department_id=14, limit=3)
# Harvest star patterns in woodwork and metalwork
# harvest_met_artifacts(query="star tile", department_id=14, limit=3)

# 3. The Hard Negative: Rosettes (Symbolic Allomorphy)
# Harvest rosettes from Near Eastern Art (Assyrian reliefs, ivory carvings)
# harvest_met_artifacts(query="rosette", department_id=3, limit=5)
# Specific search for ivory rosettes (very common in Nimrud)
# harvest_met_artifacts(query="rosette plaque", department_id=3, limit=3)

# 4. The Hard Negative: Five-Pointed Stars
# Egyptian Art (Department 10)
# harvest_met_artifacts(query="star", department_id=10, limit=5)
# General collection search for pentagrams/five-pointed stars
# harvest_met_artifacts(query="five-pointed star", limit=5)
