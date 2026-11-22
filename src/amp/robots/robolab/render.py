#
# Added by the amp-split patches, not part of Apple's ml-amp; see the accompanying LICENSE file.
#
"""Isaac Lab RenderCfg settings on every registered RoboLab env config (amp-split patch).

Isaac Sim 6's default rendering is far noisier than Isaac Sim 5's; DLSS Quality
(`dlss_mode=2`, with the driver's NGX library in the container) brings it back near 5.0's. The
fields are set after each config's own __post_init__, so before RoboLab builds the env and before
Isaac Lab applies the config to the simulation context.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MARKER = "_amp_render"


def apply_render(render: Mapping[str, Any], registry: Mapping[str, Any] | None = None) -> list[str]:
    """Wrap every registered env config class so its sim.render gets these fields; returns the
    wrapped env ids. RuntimeError when none is registered; a field RenderCfg lacks raises
    ValueError when the config is built."""
    if not render:
        return []
    if registry is None:
        import gymnasium as gym

        registry = gym.envs.registry
    fields = dict(render)
    patched, found = [], 0
    for env_id, spec in list(registry.items()):
        kwargs = getattr(spec, "kwargs", None) or {}
        base = kwargs.get("env_cfg_entry_point")
        if not isinstance(base, type):
            continue
        found += 1
        if getattr(base, MARKER, None) is not None:
            continue            # already wrapped: a second call changes nothing

        def post_init(self: Any, _base: type = base) -> None:
            post = getattr(_base, "__post_init__", None)
            if post is not None:
                post(self)
            target = self.sim.render
            for name, value in fields.items():
                if not hasattr(target, name):
                    raise ValueError(f"RenderCfg has no field {name!r}")
                setattr(target, name, value)

        wrapped = type(base.__name__, (base,), {"__post_init__": post_init, MARKER: fields})
        wrapped.__module__ = base.__module__
        wrapped.__qualname__ = getattr(base, "__qualname__", base.__name__)
        kwargs["env_cfg_entry_point"] = wrapped
        patched.append(env_id)
    if not found:
        raise RuntimeError("no RoboLab env config is registered: register the envs first")
    return patched
