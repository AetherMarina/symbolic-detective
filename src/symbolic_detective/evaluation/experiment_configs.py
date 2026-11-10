# Phase A — Individual feature ablations
PHASE_A_EXPERIMENTS = {
    "exp01_harmonics": {"harmonics": True},
    "exp02_peaks": {"peaks": True},
    "exp03_pointedness": {"pointedness": True},
    "exp04_radius": {"radius": True},
    "exp05_central_structure": {"central_structure": True},
}

# Phase B — Feature combinations
PHASE_B_EXPERIMENTS = {
    "b01_harmonics_peaks": {"harmonics": True, "peaks": True},
    "b02_peaks_pointedness": {"peaks": True, "pointedness": True},
    "b03_pointedness_radius": {"pointedness": True, "radius": True},
    "b04_harmonics_peaks_pointedness": {"harmonics": True, "peaks": True, "pointedness": True},
    "b05_all_features": {"harmonics": True, "peaks": True, "pointedness": True, "radius": True, "central_structure": True},
    "b06_harmonics_pointedness": {"harmonics": True, "pointedness": True}
}

EXPERIMENTS = {**PHASE_A_EXPERIMENTS, **PHASE_B_EXPERIMENTS}
