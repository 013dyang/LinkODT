# AI Director v0.3 — Local Qwen GGUF

## Design

AI Director is a translator, not a renderer controller.

```text
Human natural language
        ↓
Local Qwen3 GGUF
        ↓
Validated LinkNex action JSON
        ↓
Camera/action executor
        ↓
Director keyframes
        ↓
Preview / MP4
```

The GGUF model is selected from disk. No Alibaba Cloud/OpenAI API key or Base URL is used.

## Example

Human:

> 先以选中的 ROI 为中心慢慢推近 1.6 倍，然后向左绕 35 度并稍微向上抬，停 1 秒，再向右转半圈，最后拉远。

The local model may translate this into actions such as:

```json
{
  "title": "ROI reveal",
  "actions": [
    {"action":"focus_roi", "roi":"ROI_01", "duration":0.6, "transition":"smooth"},
    {"action":"push_in", "factor":1.6, "duration":1.2, "transition":"ease_in_out"},
    {"action":"orbit_left", "degrees":35, "duration":1.4, "transition":"orbit"},
    {"action":"orbit_up", "degrees":15, "duration":0.8, "transition":"orbit"},
    {"action":"hold", "duration":1.0, "transition":"hold"},
    {"action":"turntable", "degrees":180, "duration":3.0, "transition":"orbit"},
    {"action":"pull_out", "factor":1.4, "duration":1.0, "transition":"ease_out"}
  ]
}
```

The actual schema contains neutral fields for unused parameters so the local llama.cpp JSON grammar is stable.

## Action vocabulary

### Orbit / rotate

- orbit_left / orbit_right
- orbit_up / orbit_down
- orbit_yaw / orbit_pitch
- turntable
- roll_left / roll_right / roll

### Look / tilt in place

- look_left / look_right
- tilt_up / tilt_down

### Move / zoom

- push_in / pull_out
- zoom
- truck_left / truck_right
- pedestal_up / pedestal_down
- dolly_forward / dolly_backward
- pan
- move_world

### Focus / framing

- focus_roi / frame_roi / zoom_to_roi
- focus_object / frame_object
- focus_point / look_at_point
- frame_all

### Absolute camera / lens

- preset
- set_camera_position
- set_fov
- set_projection

### Timing

- hold

## Transition vocabulary

- smooth
- linear
- orbit
- ease_in
- ease_out
- ease_in_out
- hold
- cut

Large orbit and turntable actions are automatically split into intermediate camera keyframes, so a 180° or 360° camera move produces a real arc instead of collapsing to identical start/end camera states.

## 3D ROI

A selected ROI is a named 3D world-coordinate target. The local model receives the names/centers/radii of current ROIs and the currently selected ROI name. It never invents a missing ROI.

## v0.4 director language additions

The local model prompt now explicitly maps full-turn, half-turn, lay-flat/top-down, and close-up detail-scan language to deterministic LinkNex actions. The action vocabulary contains 42 validated camera/director actions.
