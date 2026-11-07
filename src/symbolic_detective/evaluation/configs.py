FEATURE_CONFIGS = {
    "raw_shape_base": {
        "harmonics": False, "peaks": False, "pointedness": False,
        "radius": False, "central_structure": False
    },
    "raw_shape_harm": {
        "harmonics": True, "peaks": False, "pointedness": False,
        "radius": False, "central_structure": False
    },
    "raw_shape_point": {
        "harmonics": False, "peaks": False, "pointedness": True,
        "radius": False, "central_structure": False
    },
}

CLASSIFICATION_FROZEN_CONFIG = {
    "feature_config": FEATURE_CONFIGS["raw_shape_harm"],
    "alpha": 0.25,
    "strategy": "weighted",
    "k": 7,
}
