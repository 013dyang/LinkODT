
# Smooth orbit / turntable v1.0

- Orbit interpolation now uses constant angular velocity instead of smoothstep easing.
- 180°/360° rotations are still split into stable arc segments, but all segments have proportional duration and no repeated slow-down at keyframe boundaries.
- Live Preview now uses Qt PreciseTimer and actual elapsed time rather than advancing one nominal frame per timer callback.
- Preview defaults to 60 FPS.
- MP4 export remains frame-accurate at the selected export FPS.
- Smooth/Ease transitions remain available for push-in, pull-out, pan and other cinematic moves.
