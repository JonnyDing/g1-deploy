__all__ = ["RetargetingModule", "extract_mimic_obs"]


def __getattr__(name: str):
    if name in __all__:
        from .core import RetargetingModule, extract_mimic_obs

        exports = {
            "RetargetingModule": RetargetingModule,
            "extract_mimic_obs": extract_mimic_obs,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
