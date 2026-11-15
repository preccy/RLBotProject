"""Record real GamePackets during an RLBot match and replay them offline through the deployment pipeline.

File format (version 2): the header line `BOTCORE-PACKETS 2\n`, then records of
(1-byte kind, uint32 little-endian length, payload):

* F  packed flat.FieldInfo
* M  UTF-8 JSON metadata (team, index, bundle hash, ...)
* P  float64 wall-clock seconds (time.perf_counter at arrival) + packed flat.GamePacket
* D  decision: int64 packet frame, int64 state tick, int32 action index, float32 policy ms, float32[obs] observation
* T  callback timing: int64 frame, float32 full-callback ms, int32 packets processed, uint8 fallback flag

A truncated final record (e.g. the process was killed) is ignored.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
from rlbot import flat

MAGIC = b"BOTCORE-PACKETS 2\n"
_DEC_HEAD = struct.Struct("<qqif")
_TIMING = struct.Struct("<qfiB")


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
        self._write(b"F", field.pack())

    def meta(self, data: dict) -> None:
        self._write(b"M", json.dumps(data).encode())

    def packet(self, packet: flat.GamePacket, wall: Optional[float] = None) -> None:
        self._write(b"P", struct.pack("<d", time.perf_counter() if wall is None else wall) + packet.pack())
        self.count += 1
        if self.count % 1200 == 0:
            self._f.flush()

    def decision(self, frame: int, tick: int, action: int, policy_ms: float, obs: np.ndarray) -> None:
        self._write(b"D", _DEC_HEAD.pack(int(frame), int(tick), int(action), float(policy_ms))
                    + np.asarray(obs, dtype="<f4").tobytes())

    def timing(self, frame: int, callback_ms: float, n_packets: int, fallback: bool) -> None:
        self._write(b"T", _TIMING.pack(int(frame), float(callback_ms), int(n_packets), int(fallback)))

    def close(self) -> None:
        self._f.close()


@dataclass
class Recording:
    field: Optional[flat.FieldInfo] = None
    meta: dict = dc_field(default_factory=dict)
    packet_wall: List[float] = dc_field(default_factory=list)
    packets: List[flat.GamePacket] = dc_field(default_factory=list)
    decision_frame: List[int] = dc_field(default_factory=list)
    decision_tick: List[int] = dc_field(default_factory=list)
    decision_action: List[int] = dc_field(default_factory=list)
    decision_ms: List[float] = dc_field(default_factory=list)
    decision_obs: List[np.ndarray] = dc_field(default_factory=list)
    timing: List[tuple] = dc_field(default_factory=list)  # (frame, callback ms, packets processed, fallback)


def load_recording(path: str | Path) -> Recording:
    data = Path(path).read_bytes()
    if not data.startswith(MAGIC):
        raise ValueError(f"{path} is not a BotCore packet recording (version 2)")
    rec = Recording()
    offset = len(MAGIC)
    while offset + 5 <= len(data):
        kind = data[offset:offset + 1]
        (length,) = struct.unpack("<I", data[offset + 1:offset + 5])
        payload = data[offset + 5:offset + 5 + length]
        if len(payload) < length:
            break
        offset += 5 + length
        if kind == b"F":
            rec.field = flat.FieldInfo.unpack(payload)
        elif kind == b"M":
            rec.meta.update(json.loads(payload.decode()))
        elif kind == b"P":
            rec.packet_wall.append(struct.unpack("<d", payload[:8])[0])
            rec.packets.append(flat.GamePacket.unpack(payload[8:]))
        elif kind == b"D":
            frame, tick, action, ms = _DEC_HEAD.unpack(payload[:_DEC_HEAD.size])
            rec.decision_frame.append(frame)
            rec.decision_tick.append(tick)
            rec.decision_action.append(action)
            rec.decision_ms.append(ms)
            rec.decision_obs.append(np.frombuffer(payload[_DEC_HEAD.size:], dtype="<f4").copy())
        elif kind == b"T":
            rec.timing.append(_TIMING.unpack(payload))
    return rec


def read_recording(path: str | Path) -> Tuple[Optional[flat.FieldInfo], Iterator[flat.GamePacket]]:
    rec = load_recording(path)
    return rec.field, iter(rec.packets)


def replay(path: str | Path, bundle_dir: str | Path, device: str = "cpu", agent_index: int = 0, log=print) -> dict:
    """Run every recorded packet through rlgym-compat + botcore reconstruction + the policy.

    Reports per-packet processing time (state update + reconstruction + obs + policy, no sockets), the number of
    non-finite observation values and the largest absolute observation value (training inputs are mostly within
    a few units; much larger values point at a unit or frame mismatch).
    """
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
