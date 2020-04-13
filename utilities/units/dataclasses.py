def __setup(mode):
  global DataClassWithDistances, distancefield
  if mode == "safe":
    from .safe.dataclasses import DataClassWithDistances, distancefield
  elif mode == "pixels":
    from .fast.pixels.dataclasses import DataClassWithDistances, distancefield
  elif mode == "microns":
    from .fast.microns.dataclasses import DataClassWithDistances, distancefield
  else:
    raise ValueError(f"Invalid mode {mode}")
