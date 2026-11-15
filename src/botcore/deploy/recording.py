"""Record real GamePackets during an RLBot match and replay them offline through the deployment pipeline.

File format: a header line `BOTCORE-PACKETS 1\\n` followed by records of
(uint32 little-endian length, packed flat.GamePacket bytes). FieldInfo is stored as the first record with a
length prefix and a marker byte so a recording is self-contained.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
from rlbot import flat

MAGIC = b"BOTCORE-PACKETS 1\n"
_KIND_FIELD = b"F"
_KIND_PACKET = b"P"


class PacketRecorder:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "wb")
        self._f.write(MAGIC)
        self.count = 0

    def _write(self, kind: bytes, payload: bytes) -> None:
        self._f.write(kind + struct.pack("<I", len(payload)) + payload)

    def field_info(self, field: flat.FieldInfo) -> None:
        self._write(_KIND_FIELD, field.pack())

    def packet(self, packet: flat.GamePacket) -> None:
        self._write(_KIND_PACKET, packet.pack())
        self.count += 1
        if self.count % 1200 == 0:
            self._f.flush()

    def close(self) -> None:
        self._f.close()


def read_recording(path: str | Path) -> Tuple[Optional[flat.FieldInfo], Iterator[flat.GamePacket]]:
    data = Path(path).read_bytes()
    if not data.startswith(MAGIC):
        raise ValueError(f"{path} is not a BotCore packet recording")
    offset = len(MAGIC)
    field = None
    packets = []
    while offset + 5 <= len(data):
        kind = data[offset:offset + 1]
        (length,) = struct.unpack("<I", data[offset + 1:offset + 5])
        payload = data[offset + 5:offset + 5 + length]
        if len(payload) < length:
            break  # truncated tail (recording interrupted); ignore the partial record
        offset += 5 + length
        if kind == _KIND_FIELD:
            field = flat.FieldInfo.unpack(payload)
        elif kind == _KIND_PACKET:
            packets.append(flat.GamePacket.unpack(payload))
    return field, iter(packets)


def replay(path: str | Path, bundle_dir: str | Path, device: str = "cpu", agent_index: int = 0, log=print) -> dict:
    """Run every recorded packet through rlgym-compat + botcore reconstruction + the policy.

    Reports per-packet processing time (state update + reconstruction + obs + policy, no sockets), the number of
    non-finite observation values and the largest absolute observation value (training inputs are mostly within
    a few units; much larger values point at a unit or frame mismatch).
    """
    import time

    from botcore.deploy.packets import CompatObsPipeline
    from botcore.export import PolicyRunner, load_bundle

    field, packets = read_recording(path)
    if field is None:
        raise ValueError("recording has no FieldInfo record")
    runner = PolicyRunner(load_bundle(bundle_dir, device=device), device=device)
    pipe = CompatObsPipeline(field)
    times, observations = [], []
    player_id = None
    active = (flat.MatchPhase.Kickoff, flat.MatchPhase.Active)
    for packet in packets:
        if not packet.players:
            continue
        if player_id is None:
            player_id = packet.players[min(agent_index, len(packet.players) - 1)].player_id
        t0 = time.perf_counter()
        pipe.update(packet)
        if packet.match_info.match_phase in active:
            obs = pipe.obs(player_id)
            runner.act(obs)
            observations.append(obs)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.asarray(times)
    obs_arr = np.asarray(observations) if observations else np.zeros((0, 1))
    report = {
        "packets": int(len(arr)),
        "decisions": int(len(observations)),
        "ms_p50": float(np.median(arr)) if len(arr) else None,
        "ms_p99": float(np.percentile(arr, 99)) if len(arr) else None,
        "ms_max": float(arr.max()) if len(arr) else None,
        "non_finite_obs": int((~np.isfinite(obs_arr)).sum()),
        "obs_abs_max": float(np.abs(obs_arr).max()) if obs_arr.size else None,
    }
    log(report)
    return report
