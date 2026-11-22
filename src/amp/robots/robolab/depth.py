#
# Added by the amp-split patches, not part of Apple's ml-amp; see the accompanying LICENSE file.
#
"""A depth-rendering variant of a RoboLab config class, built without instantiating it.

RoboLab's `variations.camera.with_depth` instantiates the class to find its cameras. On Isaac Lab
3, instantiating `DroidCfg` fails: its `table_fixture` is a frozen dataclass, and Isaac Lab 3's
config post-init rewrites every nested dataclass in place. RoboLab's own env building never
instantiates `DroidCfg` directly (the scene config overrides `table_fixture` first), so only the
depth variant hits this. This reads the cameras from the class's dataclass fields instead.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable
from typing import Any


def field_default(f: dataclasses.Field) -> Any:
    """A field's default value, or dataclasses.MISSING."""
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return dataclasses.MISSING


def with_depth_cls(cfg_cls: type, camera_cls: type, configclass: Callable[[type], type]) -> type:
    """As RoboLab's with_depth: a subclass whose cameras (fields of `camera_cls`) also render
    depth, under the name <Cls>WithDepth and processed by `configclass`; the class itself when it
    has no camera without depth. The class is not instantiated."""
    overrides = {}
    for name, f in getattr(cfg_cls, "__dataclass_fields__", {}).items():
        value = field_default(f)
        if isinstance(value, camera_cls) and "depth" not in value.data_types:
            camera = copy.deepcopy(value)
            camera.data_types = [*camera.data_types, "depth"]
            overrides[name] = camera
    if not overrides:
        return cfg_cls
    variant = type(f"{cfg_cls.__name__}WithDepth", (cfg_cls,), overrides)
    variant.__module__ = cfg_cls.__module__
    return configclass(variant)


def robolab_with_depth(cfg_cls: type) -> type:
    """RoboLab's with_depth, falling back to with_depth_cls when instantiating the class fails on
    a frozen nested dataclass (Isaac Lab 3); Isaac Lab 2 never takes the fallback."""
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass
    from robolab.variations.camera import with_depth

    try:
        return with_depth(cfg_cls)
    except dataclasses.FrozenInstanceError:
        return with_depth_cls(cfg_cls, CameraCfg, configclass)
