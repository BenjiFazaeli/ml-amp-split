#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Local GPU selection for RoboLab Kits: busy rule, UUID pinning and slots per GPU."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

GIB = 1024**3
MIB = 1024**2
MAX_USED_GIB = 4.0
# A Kit's VRAM for the slot bound: "about 9 GB" in validation (docs/quickstart.md) plus 2 GiB, so
# two Kits fit a 22.5 GiB RTX 6000 (amp-split patch 0012; was 12.0, which allowed one).
KIT_VRAM_GIB = 11.0
KIT_RSS_GIB = 18.0
HOST_RESERVE_GIB = 16.0
AUTO_MAX_SLOTS = 1
# Kits per GPU that an explicit --slots-per-gpu may ask for (amp-split patch 0012); auto stays 1.
MAX_SLOTS = 2
SMALL_WARNING = ("warning: GPU {index} {name} has {gib:.1f} GiB of VRAM; a Kit used about 9 GB in "
                 "validation and may not fit; pass --gpus to leave it out")
QUERY = ("--query-gpu=index,uuid,name,memory.total,memory.used", "--format=csv,noheader,nounits")
# MIG slices (amp-split patch): --query-gpu lists only the parent GPUs, with the whole card's
# memory, so slices come from `nvidia-smi -L`, e.g.
#   GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition (UUID: GPU-2f75...)
#     MIG 2g.48gb     Device  0: (UUID: MIG-49a9...)
LIST = ("-L",)
GPU_LINE = re.compile(r"^GPU\s+(\d+):\s*(.*?)\s*\(UUID:\s*(GPU-[^)\s]+)\)")
MIG_LINE = re.compile(r"^\s+MIG\s+(\S+)\s+Device\s+(\d+):\s*\(UUID:\s*(MIG-[^)\s]+)\)")
MIG_PROFILE_GB = re.compile(r"\.(\d+)gb\b")


class GpuError(RuntimeError):
    """No usable GPU, or a requested GPU or slot count the rules refuse."""


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    name: str
    total_mib: int
    used_mib: int
    # A MIG slice (amp-split patch): uuid is its MIG- UUID, index its position among the slices,
    # total_mib the profile's memory, used_mib 0 (a slice is the job's own; nvidia-smi does not
    # report its use). Kit takes the slice by CUDA_VISIBLE_DEVICES alone, with no activeGpu.
    mig: bool = False

    @property
    def total_gib(self) -> float:
        return self.total_mib * MIB / GIB

    @property
    def used_gib(self) -> float:
        return self.used_mib * MIB / GIB

    @property
    def small(self) -> bool:
        return self.total_gib < KIT_VRAM_GIB

    def small_warning(self) -> str:
        return SMALL_WARNING.format(index=self.index, name=self.name, gib=self.total_gib)

    @property
    def busy(self) -> bool:
        return self.used_gib >= MAX_USED_GIB

    def refusal(self) -> str | None:
        """Why this GPU may not host a Kit, or None: only a busy GPU is refused."""
        if self.busy:
            return f"{self.used_gib:.1f} GiB already in use (>= {MAX_USED_GIB:g} GiB)"
        return None

    def describe(self) -> str:
        if self.mig:
            return f"MIG slice {self.index} {self.name} {self.uuid}: {self.total_gib:.1f} GiB"
        return (f"GPU {self.index} {self.name} {self.uuid}: {self.total_gib:.1f} GiB, "
                f"{self.used_gib:.1f} GiB used")


def parse_smi(text: str) -> list[Gpu]:
    """nvidia-smi's CSV (QUERY) as Gpus, in index order."""
    out = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5 or not parts[0].isdigit():
            continue
        try:
            out.append(Gpu(int(parts[0]), parts[1], parts[2], int(float(parts[3])),
                           int(float(parts[4]))))
        except ValueError:
            continue
    return sorted(out, key=lambda g: g.index)


def parse_mig(text: str) -> list[Gpu]:
    """The MIG slices in `nvidia-smi -L` output, in listed order (amp-split patch); [] for none."""
    out: list[Gpu] = []
    parent = ""
    for line in text.splitlines():
        if m := GPU_LINE.match(line):
            parent = m.group(2)
            continue
        if not (m := MIG_LINE.match(line)):
            continue
        profile, uuid = m.group(1), m.group(3)
        gb = MIG_PROFILE_GB.search(profile)
        total_mib = int(int(gb.group(1)) * 1e9 / MIB) if gb else 0
        out.append(Gpu(len(out), uuid, f"{parent} MIG {profile}".strip(), total_mib, 0, mig=True))
    return out


def choose_devices(gpus: Sequence[Gpu], slices: Sequence[Gpu],
                   visible: str | None = None) -> list[Gpu]:
    """The devices a Kit may use: the MIG slices when there are any (their parents cannot host
    a Kit, and report other jobs' use), restricted to CUDA_VISIBLE_DEVICES' MIG UUIDs when it
    names some; else the GPUs, as before (amp-split patch)."""
    if not slices:
        return list(gpus)
    named = {v.strip() for v in (visible or "").split(",") if v.strip().startswith("MIG-")}
    chosen = [s for s in slices if not named or s.uuid in named]
    return [Gpu(i, s.uuid, s.name, s.total_mib, s.used_mib, mig=True)
            for i, s in enumerate(chosen)]


def _smi(exe: str, args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout_s,  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-tainted-env-args.dangerous-subprocess-use-tainted-env-args
                              check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise GpuError(f"nvidia-smi: {type(e).__name__}: {e}") from e


def query(smi: str | None = None, *, timeout_s: float = 30.0) -> list[Gpu]:
    """Every GPU nvidia-smi reports, or this job's MIG slices on a MIG node; GpuError when
    nvidia-smi is missing or fails."""
    exe = smi or os.environ.get("AMP_NVIDIA_SMI") or shutil.which("nvidia-smi")
    if not exe:
        raise GpuError("nvidia-smi not found")
    p = _smi(exe, QUERY, timeout_s)
    if p.returncode != 0:
        raise GpuError(f"nvidia-smi exited {p.returncode}: {p.stderr.strip()[:300]}")
    gpus = parse_smi(p.stdout)
    try:
        listed = _smi(exe, LIST, timeout_s)
    except GpuError:
        return gpus
    slices = parse_mig(listed.stdout) if listed.returncode == 0 else []
    return choose_devices(gpus, slices, os.environ.get("CUDA_VISIBLE_DEVICES"))


def select(spec: str, gpus: Sequence[Gpu], *,
           warn: Callable[[str], None] = lambda m: print(m, file=sys.stderr)
           ) -> tuple[list[Gpu], list[tuple[Gpu, str]]]:
    """(chosen, refused with why) for --gpus auto or a list of indices or UUIDs."""
    refused = [(g, why) for g in gpus if (why := g.refusal()) is not None]
    if spec.strip() in ("", "auto"):
        chosen = [g for g in gpus if g.refusal() is None]
        if not chosen:
            if not gpus:
                raise GpuError("no NVIDIA GPU found (nvidia-smi lists none)")
            raise GpuError(f"every GPU is busy (>= {MAX_USED_GIB:g} GiB in use): "
                           + "; ".join(f"{g.describe()} ({why})" for g, why in refused))
        _warn_chosen(chosen, warn)
        return chosen, refused
    by_key = {str(g.index): g for g in gpus} | {g.uuid: g for g in gpus}
    chosen = []
    for key in (k.strip() for k in spec.split(",") if k.strip()):
        g = by_key.get(key)
        if g is None:
            raise GpuError(f"--gpus {key}: no such GPU ({', '.join(map(str, by_key))})")
        if (why := g.refusal()) is not None:
            raise GpuError(f"--gpus {key}: refused, {g.describe()}: {why}")
        if g not in chosen:
            chosen.append(g)
    _warn_chosen(chosen, warn)
    return chosen, refused


def _warn_chosen(chosen: Sequence[Gpu], warn: Callable[[str], None]) -> None:
    for g in chosen:
        if g.small:
            warn(g.small_warning())


def host_ram_bytes() -> int:
    """Physical memory of this host."""
    if sys.platform.startswith("linux"):
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError):
        return 0


def ram_bound(n_gpus: int, ram_bytes: int) -> int:
    """The most Kits per GPU that host RAM holds."""
    room = ram_bytes / GIB - HOST_RESERVE_GIB
    return max(0, int(room // (KIT_RSS_GIB * max(1, n_gpus))))


def slots_per_gpu(spec: str, gpus: Sequence[Gpu], *, ram_bytes: int | None = None) -> int:
    """Kits per GPU for --slots-per-gpu auto|1..MAX_SLOTS, bounded by host RAM and the smallest GPU's VRAM (never below one); auto picks at most AUTO_MAX_SLOTS."""
    ram = host_ram_bytes() if ram_bytes is None else ram_bytes
    by_ram = ram_bound(len(gpus), ram)
    by_vram = min(max(1, int(g.total_gib // KIT_VRAM_GIB)) for g in gpus) if gpus else 0
    bound = min(by_ram, by_vram)
    if spec.strip() in ("", "auto"):
        n = min(AUTO_MAX_SLOTS, bound)
        if n < 1:
            raise GpuError(f"not enough host RAM for one Kit per GPU ({ram / GIB:.0f} GiB, "
                           f"{len(gpus)} GPUs)")
        return n
    try:
        n = int(spec)
    except ValueError:
        raise GpuError(f"--slots-per-gpu {spec!r} is not auto or an integer") from None
    if n < 1 or n > MAX_SLOTS:
        raise GpuError(f"--slots-per-gpu must be auto or 1..{MAX_SLOTS}")
    if n > bound:
        raise GpuError(f"--slots-per-gpu {n} exceeds what this host holds ({bound}: {by_ram} by "
                       f"{ram / GIB:.0f} GiB RAM at {KIT_RSS_GIB:g} GiB per Kit, {by_vram} by "
                       f"VRAM at {KIT_VRAM_GIB:g} GiB per Kit)")
    return n
