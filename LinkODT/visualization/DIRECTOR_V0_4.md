# LinkNex Visualization Studio v0.4 — Director improvements

## Formal preview / movie

3D ROI helper spheres are editing guides only. They are automatically hidden during:

- Director Preview playback
- MP4 export

They are restored after preview/export finishes.

## Natural-language directing

Local Qwen explicitly understands common directing phrases such as:

- 转一圈 / 绕一圈 / full revolution -> 360° turntable
- 半圈 -> 180° turntable
- 四分之一圈 -> 90° turntable
- generic orbit/turn with no angle -> 90° by default
- 稍微转一下 -> 30–45°
- 平躺 / 平放 -> top-down lay-flat view
- 拉近后平移看细节 -> focus/frame target, push in, then detail scan

## Quick director presets

Four one-click presets create a camera timeline and start preview:

1. 360° Hero Orbit
2. Push-in Detail Scan
3. Cinematic Arc
4. ROI Tour

If a selected/defined ROI exists, the presets use ROI focus points. ROI Tour visits every ROI. If no ROI exists, the whole visible scene is used.

## Manual camera toolbar

The small camera rotations were replaced by larger directorial moves:

- Orbit left/right/up/down: 90°
- Turn: 180°
- Roll left/right: 90°
- Zoom: 1.5× in/out
