# Intensity brightness mapping v1.1

Applies only to XYZ + Intensity point clouds.

- `DEFAULT_INTENSITY_MIN_BRIGHTNESS = 0.35` controls the darkest displayed intensity.
  Raise it (for example to `0.45`) to make the organelle shell brighter.
- The selected Object Color is now a reference/base hue, not the 100% endpoint.
- The 100% endpoint is the brightest possible RGB with the same hue, computed by
  proportionally scaling the base RGB until one channel reaches 1.0.
- RGB instance/PCD colors, meshes, and other formats are unchanged.
