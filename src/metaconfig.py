metaconfig = {
    "samples": {
        "discordantThreshold": 0.01,
        "accurateThreshold": 0.85,
        "sequester": {
            "discordant": {
                "case": False,
                "control": False,
                "random": [], # case, control, or both
            },
            "accurate": {
                "case": True,
                "control": False,
                "random": [], # case, control, or both
            },
        },
    },
    "tracking": {"lastIteration": 0},
}
