__all__ = ["BVHInputProvider", "Pico4InputProvider", "UDPBVHInputProvider"]


def __getattr__(name: str):
    if name == "BVHInputProvider":
        from deploy.inputs.bvh_provider import BVHInputProvider

        return BVHInputProvider
    if name == "Pico4InputProvider":
        from deploy.inputs.pico4_provider import Pico4InputProvider

        return Pico4InputProvider
    if name == "UDPBVHInputProvider":
        from deploy.inputs.udp_bvh_provider import UDPBVHInputProvider

        return UDPBVHInputProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
