"""Realtime UDP BVH streaming input provider.

Receives BVH motion data over UDP (one packet = one frame of
whitespace-separated floats) and converts each frame to a HumanFrame
dict using the same processing logic as the offline BVH provider.

The realtime skeleton definitions are hardcoded — no reference BVH needed.
"""
from __future__ import annotations

import logging
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from deploy.inputs.bvh_provider import process_single_bvh_frame
from deploy.inputs.realtime_frame_cache import RealtimeFrameCache
from deploy.inputs.realtime_packet import (
    ControlEvent,
    HumanFrame,
    RealtimeInputPacket,
)
from deploy.inputs.rot_utils import matrix_to_quat_np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SkeletonDef:
    bone_names: list[str]
    bone_parents: NDArray[np.int32]
    offsets: NDArray[np.float64]
    euler_order: str
    channels: int
    scale_divisor: float

# Axis Studio/Noitom 59-joint BVH body + hands skeleton. Offsets are in
# centimeters, matching offline BVH processing.
_NOITOM_BONE_NAMES: list[str] = [
    "Hips",
    "RightUpLeg", "RightLeg", "RightFoot",
    "LeftUpLeg", "LeftLeg", "LeftFoot",
    "Spine", "Spine1", "Spine2", "Neck", "Neck1", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3",
    "RightInHandIndex",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3",
    "RightInHandMiddle",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
    "RightInHandRing",
    "RightHandRing1", "RightHandRing2", "RightHandRing3",
    "RightInHandPinky",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3",
    "LeftInHandIndex",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3",
    "LeftInHandMiddle",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3",
    "LeftInHandRing",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
    "LeftInHandPinky",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3",
]

_NOITOM_BONE_PARENTS: list[int] = [
    -1,
    0, 1, 2,
    0, 4, 5,
    0, 7, 8, 9, 10, 11,
    9, 13, 14, 15,
    16, 17, 18,
    16, 20, 21, 22,
    16, 24, 25, 26,
    16, 28, 29, 30,
    16, 32, 33, 34,
    9, 36, 37, 38,
    39, 40, 41,
    39, 43, 44, 45,
    39, 47, 48, 49,
    39, 51, 52, 53,
    39, 55, 56, 57,
]

# fmt: off
_NOITOM_OFFSETS: list[list[float]] = [
    [0.0, 91.43, 0.0],
    [-9.25, 0.0, 0.0], [0.0, -41.869999, 0.0], [0.0, -41.869999, 0.0],
    [9.25, 0.0, 0.0], [0.0, -41.869999, 0.0], [0.0, -41.869999, 0.0],
    [0.0, 7.818, 0.0], [0.0, 17.309999, 0.0], [0.0, 12.285, 0.0],
    [0.0, 18.427, 0.0], [0.0, 4.735, 0.0], [0.0, 4.735, 0.0],
    [-2.792, 12.843, 0.0], [-13.208, 0.0, 0.0], [-26.5, 0.0, 0.0], [-26.0, 0.0, 0.0],
    [-1.842, -0.461, 2.395], [-3.682, 0.0, 0.0], [-2.559, 0.0, 0.0],
    [-3.224, 0.508, 1.978], [-5.217, -0.091, 0.999], [-3.62, 0.0, 0.0], [-2.052, 0.0, 0.0],
    [-3.382, 0.518, 0.757], [-5.174, -0.084, 0.314], [-3.949, 0.0, 0.0], [-2.476, 0.0, 0.0],
    [-3.366, 0.538, -0.129], [-4.635, -0.022, -0.479], [-3.442, 0.0, 0.0], [-2.388, 0.0, 0.0],
    [-3.161, 0.47, -1.202], [-4.141, -0.022, -1.091], [-2.757, 0.0, 0.0], [-1.742, 0.0, 0.0],
    [2.792, 12.843, 0.0], [13.208, 0.0, 0.0], [26.5, 0.0, 0.0], [26.0, 0.0, 0.0],
    [1.842, -0.461, 2.395], [3.682, 0.0, 0.0], [2.559, 0.0, 0.0],
    [3.224, 0.508, 1.978], [5.217, -0.091, 0.999], [3.62, 0.0, 0.0], [2.052, 0.0, 0.0],
    [3.382, 0.518, 0.757], [5.174, -0.084, 0.314], [3.949, 0.0, 0.0], [2.476, 0.0, 0.0],
    [3.366, 0.538, -0.129], [4.635, -0.022, -0.479], [3.442, 0.0, 0.0], [2.388, 0.0, 0.0],
    [3.161, 0.47, -1.202], [4.141, -0.022, -1.091], [2.757, 0.0, 0.0], [1.742, 0.0, 0.0],
]
# fmt: on

NOITOM_SKELETON = _SkeletonDef(
    bone_names=_NOITOM_BONE_NAMES,
    bone_parents=np.array(_NOITOM_BONE_PARENTS, dtype=np.int32),
    offsets=np.array(_NOITOM_OFFSETS, dtype=np.float64),
    euler_order="zyx",
    channels=3,
    scale_divisor=100.0,
)

_SKELETONS: dict[str, _SkeletonDef] = {
    "noitom": NOITOM_SKELETON,
}


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------

def _lerp_frames(frame_a: HumanFrame, frame_b: HumanFrame, alpha: float) -> HumanFrame:
    """Linearly interpolate two HumanFrame dicts (position lerp, quat slerp)."""
    from deploy.retargeting.gmr.utils.lafan_vendor import utils

    result: HumanFrame = {}
    for bone in frame_b:
        if bone not in frame_a:
            result[bone] = frame_b[bone]
            continue
        pos_a, quat_a = frame_a[bone]
        pos_b, quat_b = frame_b[bone]
        pos = pos_a * (1.0 - alpha) + pos_b * alpha
        quat = utils.quat_slerp(quat_a, quat_b, alpha)
        result[bone] = (pos, quat)
    return result


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class UDPBVHInputProvider:
    """Realtime input provider that receives BVH frames over UDP.

    Implements the ``RealtimeInputProvider`` protocol.
    """

    def __init__(
        self,
        bvh_format: str = "noitom",
        human_height: float = 1.75,
        udp_host: str = "",
        udp_port: int = 1118,
        udp_timeout: float = 30.0,
        buffer_size: int = 60,
    ) -> None:
        skel = _SKELETONS.get(bvh_format)
        if skel is None:
            raise ValueError(
                f"Unsupported bvh_format '{bvh_format}' for UDP streaming. "
                f"Supported: {list(_SKELETONS)}."
            )

        self._bone_names = skel.bone_names
        self._bone_parents = skel.bone_parents
        self._offsets = skel.offsets.copy()
        self._euler_order = skel.euler_order
        self._channels = skel.channels
        self._scale_divisor = skel.scale_divisor

        N = len(self._bone_names)
        if self._channels == 3:
            self._expected_floats = 3 + N * 3
        elif self._channels == 6:
            self._expected_floats = N * 6
        else:
            raise ValueError(f"Unsupported channel count: {self._channels}")

        # Coordinate transform: Y-up → Z-up
        self._rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        self._rotation_quat = matrix_to_quat_np(self._rotation_matrix, scalar_first=True)

        self._bvh_format = bvh_format
        self._human_format = bvh_format
        self._human_height = human_height
        self._udp_host = udp_host
        self._udp_port = udp_port
        self._udp_timeout = udp_timeout

        self._cache = RealtimeFrameCache[HumanFrame](buffer_size=buffer_size)
        self._lock = threading.Lock()
        self._first_frame_event = threading.Event()
        self._running = True
        self._control_events: deque[ControlEvent] = deque()

        # Bind socket on main thread so port/address errors surface immediately.
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(2.0)
        self._sock.bind((self._udp_host, self._udp_port))

        self._thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._thread.start()
        log.info("UDPBVHInputProvider listening on %s:%d", udp_host or "0.0.0.0", udp_port)

    # -- properties --

    @property
    def fps(self) -> float:
        return self._cache.fps()

    @property
    def bone_names(self) -> list[str]:
        return self._bone_names

    @property
    def bone_parents(self) -> NDArray[np.int32]:
        return self._bone_parents

    @property
    def human_format(self) -> str:
        return self._human_format

    @property
    def human_height(self) -> float:
        return self._human_height

    # -- InputProvider --

    def is_available(self) -> bool:
        return self._running and self._thread.is_alive()

    def get_frame(self) -> HumanFrame:
        self._first_frame_event.wait(timeout=self._udp_timeout)
        if not self._first_frame_event.is_set():
            raise TimeoutError(
                f"No UDP BVH data received within {self._udp_timeout}s on port {self._udp_port}"
            )
        with self._lock:
            return self._cache.latest()

    # -- RealtimeInputProvider --

    def get_frame_packet(self) -> tuple[HumanFrame, float, int]:
        self._first_frame_event.wait(timeout=self._udp_timeout)
        if not self._first_frame_event.is_set():
            raise TimeoutError(
                f"No UDP BVH data received within {self._udp_timeout}s on port {self._udp_port}"
            )
        with self._lock:
            return self._cache.latest_packet()

    def get_realtime_input_packet(self) -> RealtimeInputPacket[HumanFrame]:
        frame, ts, seq = self.get_frame_packet()
        with self._lock:
            events = tuple(self._control_events)
            self._control_events.clear()
        return RealtimeInputPacket(frame=frame, timestamp_s=ts, seq=seq, control_events=events)

    def sample_frame(self, query_time_s: float, delay_s: float) -> HumanFrame:
        """Return an interpolated frame for the requested time."""
        with self._lock:
            snap = self._cache.snapshot()

        if not snap:
            return self.get_frame()

        target = query_time_s - delay_s

        if len(snap) < 2:
            return snap[0][0]

        for i in range(len(snap) - 1, 0, -1):
            ts_b = snap[i][1]
            ts_a = snap[i - 1][1]
            if ts_a <= target <= ts_b:
                dt = ts_b - ts_a
                alpha = (target - ts_a) / dt if dt > 0 else 1.0
                return _lerp_frames(snap[i - 1][0], snap[i][0], alpha)

        if target <= snap[0][1]:
            return snap[0][0]
        return snap[-1][0]

    def close(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)
        log.info("UDPBVHInputProvider closed")

    # -- receiver thread --

    def _receiver_loop(self) -> None:
        try:
            while self._running:
                try:
                    data, _addr = self._sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError:
                    if self._running:
                        log.warning("UDP socket error, stopping receiver")
                    break

                self._process_packet(data)
        finally:
            self._sock.close()

    def _process_packet(self, data: bytes) -> None:
        try:
            text = data.decode("utf-8").strip()
            if not text:
                return
            floats = np.fromstring(text, sep=" ", dtype=np.float64)
        except (UnicodeDecodeError, ValueError) as exc:
            log.warning("Malformed UDP packet: %s", exc)
            return

        if len(floats) != self._expected_floats:
            log.warning(
                "Expected %d floats, got %d — skipping frame",
                self._expected_floats,
                len(floats),
            )
            return

        frame = process_single_bvh_frame(
            data_floats=floats,
            offsets=self._offsets,
            bone_names=self._bone_names,
            bone_parents=self._bone_parents,
            euler_order=self._euler_order,
            rotation_quat=self._rotation_quat,
            rotation_matrix=self._rotation_matrix,
            format=self._bvh_format,
            scale_divisor=self._scale_divisor,
            channels=self._channels,
        )

        now = time.monotonic()
        with self._lock:
            self._cache.append(frame, now)

        if not self._first_frame_event.is_set():
            self._first_frame_event.set()
            log.info("First UDP BVH frame received")
