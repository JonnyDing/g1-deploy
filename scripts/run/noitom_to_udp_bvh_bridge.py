"""Bridge Axis Studio/Noitom BVH text frames to g1-deploy UDP BVH frames.

The bridge listens for a BVH-style stream from Axis Studio, normalizes each
payload to the 180-float frame expected by UDPBVHInputProvider, and forwards it
to the G1 control computer.
"""

from __future__ import annotations

import argparse
import logging
import re
import socket
import sys
from collections.abc import Sequence


FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def extract_floats(payload: bytes) -> list[float]:
    """Extract decimal/scientific-notation floats from an Axis text payload."""
    text = payload.decode("utf-8", errors="ignore")
    return [float(match.group(0)) for match in FLOAT_RE.finditer(text)]


def normalize_frame(
    values: Sequence[float],
    *,
    expected_floats: int = 180,
    trim: str = "first",
) -> list[float]:
    """Normalize a raw Axis frame to the UDPBVHInputProvider float count.

    Supported inputs:
    - 180 floats: forwarded unchanged.
    - 181 floats: first value is treated as a frame counter and dropped.
    - 177 floats: root translation is missing, so [0, 0, 0] is prepended.
    - More than 180 floats: trimmed from the front or back, controlled by trim.
    """
    frame = [float(value) for value in values]
    count = len(frame)

    if count == expected_floats:
        return frame
    if count == expected_floats + 1:
        return frame[1:]
    if count == expected_floats - 3:
        return [0.0, 0.0, 0.0, *frame]
    if count > expected_floats:
        if trim == "last":
            return frame[-expected_floats:]
        if trim == "first":
            return frame[:expected_floats]
        raise ValueError(f"Invalid trim mode: {trim}")

    raise ValueError(f"Expected at least {expected_floats - 3} floats, got {count}")


def encode_frame(values: Sequence[float]) -> bytes:
    """Encode one normalized frame as UTF-8 whitespace-separated floats."""
    return " ".join(f"{value:.9g}" for value in values).encode("utf-8")


def run_bridge(args: argparse.Namespace) -> int:
    log = logging.getLogger("noitom_bridge")
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind((args.listen_host, args.listen_port))

    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    destination = (args.destination_host, args.destination_port)

    forwarded = 0
    dropped = 0
    log.info(
        "Listening for Axis/Noitom BVH UDP on %s:%d, forwarding to %s:%d",
        args.listen_host or "0.0.0.0",
        args.listen_port,
        args.destination_host,
        args.destination_port,
    )

    try:
        while True:
            payload, source = rx.recvfrom(args.recv_buffer)
            values = extract_floats(payload)
            try:
                frame = normalize_frame(
                    values,
                    expected_floats=args.expected_floats,
                    trim=args.trim,
                )
            except ValueError as exc:
                dropped += 1
                log.warning("Dropping packet from %s:%d: %s", source[0], source[1], exc)
                continue

            tx.sendto(encode_frame(frame), destination)
            forwarded += 1

            if args.once:
                log.info("Forwarded one frame; exiting because --once was set")
                return 0
            if forwarded % args.log_every == 0:
                log.info("Forwarded %d frames, dropped %d", forwarded, dropped)
    except KeyboardInterrupt:
        log.info("Interrupted after forwarding %d frames, dropped %d", forwarded, dropped)
        return 0
    finally:
        rx.close()
        tx.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forward Axis Studio/Noitom BVH text UDP frames to g1-deploy UDP BVH input."
    )
    parser.add_argument("--listen-host", default="0.0.0.0", help="Local host/IP for Axis BVH UDP.")
    parser.add_argument("--listen-port", type=int, default=7012, help="Local Axis BVH UDP port.")
    parser.add_argument("--destination-host", required=True, help="G1 control computer IP/host.")
    parser.add_argument("--destination-port", type=int, default=1118, help="UDPBVHInputProvider port.")
    parser.add_argument("--expected-floats", type=int, default=180, help="Output floats per frame.")
    parser.add_argument(
        "--trim",
        choices=("first", "last"),
        default="first",
        help="How to trim Axis packets that contain more than expected-floats values.",
    )
    parser.add_argument("--recv-buffer", type=int, default=65535, help="UDP receive buffer size.")
    parser.add_argument("--log-every", type=int, default=300, help="Frame interval for progress logs.")
    parser.add_argument("--once", action="store_true", help="Forward one valid frame then exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    return run_bridge(args)


if __name__ == "__main__":
    sys.exit(main())
