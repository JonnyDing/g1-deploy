"""Per-step latency recorder for sim2real teleoperation profiling.

Records six timestamps per policy step to CSV for offline analysis:

    T0  – pico_bridge receive_time_s (packet.timestamp_s)
    T1  – GMR retargeting + timeline append complete
    T2  – reference window sampled (yaw aligned + smoothed)
    T3  – 166D observation built
    T4  – ONNX inference + get_target_dof_pos complete
    T5  – SDK send_positions() complete

All timestamps use ``time.monotonic()`` except T0 which comes from pico_bridge.
Both share the same clock base (``time.monotonic``) and are directly comparable.

Usage::

    recorder = LatencyTraceRecorder(output_path)
    recorder.open()
    ...
    recorder.record_step(
        t_frame_arrival=packet.timestamp_s,
        t_retarget_done=...,
        t_ref_sampled=...,
        t_obs_built=...,
        t_action_done=...,
        t_cmd_sent=...,
        frame_seq=...,
        is_new_frame=...,
    )
    recorder.close()
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_COLUMNS = (
    "step",
    "t_frame_arrival",
    "t_retarget_done",
    "t_ref_sampled",
    "t_obs_built",
    "t_action_done",
    "t_cmd_sent",
    "frame_seq",
    "is_new_frame",
)


class LatencyTraceRecorder:
    """CSV-based per-step latency recorder for sim2real profiling."""

    def __init__(self, output_path: str | Path) -> None:
        self._path = Path(output_path)
        self._file: object | None = None
        self._writer: object | None = None
        self._step: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._file is not None:
            raise RuntimeError("LatencyTraceRecorder already open")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(str(self._path), "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(_COLUMNS)
        self._step = 0
        logger.info("LatencyTraceRecorder opened: %s", self._path)

    def close(self) -> None:
        if self._file is None:
            return
        self._file.close()
        self._file = None
        self._writer = None
        logger.info("LatencyTraceRecorder closed (%d steps)", self._step)

    def __enter__(self) -> LatencyTraceRecorder:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        *,
        t_frame_arrival: float,
        t_retarget_done: float,
        t_ref_sampled: float,
        t_obs_built: float,
        t_action_done: float,
        t_cmd_sent: float,
        frame_seq: int,
        is_new_frame: bool,
    ) -> None:
        if self._writer is None:
            raise RuntimeError("LatencyTraceRecorder not open")
        self._writer.writerow((
            self._step,
            f"{t_frame_arrival:.9f}",
            f"{t_retarget_done:.9f}",
            f"{t_ref_sampled:.9f}",
            f"{t_obs_built:.9f}",
            f"{t_action_done:.9f}",
            f"{t_cmd_sent:.9f}",
            frame_seq,
            int(is_new_frame),
        ))
        self._step += 1
