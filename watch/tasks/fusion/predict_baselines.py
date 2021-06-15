from types import SimpleNamespace

datasets = {
    "onera": "OneraCD_2018",
    "drop0_s2": "Drop0AlignMSI_S2",
}

methods = {
    "Axial": "AxialTransformerChangeDetector",
    # "Joint": "JointTransformerChangeDetector",
    "SpaceTimeMode": "SpaceTimeModeTransformerChangeDetector",
    "SpaceMode": "SpaceModeTransformerChangeDetector",
    "SpaceTime": "SpaceTimeTransformerChangeDetector",
    "TimeMode": "TimeModeTransformerChangeDetector",
    "Space": "SpaceTransformerChangeDetector",
}
