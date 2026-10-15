# LinkNex Visualization Studio v0.1

## Purpose

Standalone 3D scene viewer and camera-movie director launched from LinkNex Module 6.

## Implemented in v0.1

### Geometry input

- OBJ mesh
- TXT / CSV point cloud
- Drag and drop
- Multiple objects in one scene
- Automatic detection of common LinkNex RGB / normal column layouts

### 3D appearance

- Per-object visibility
- Object color
- Edge / wireframe line color
- File RGB on/off
- Surface
- Surface + edges
- Wireframe
- Points
- Opacity
- Point size
- Line width
- Surface / PCD normal arrows
- Normal length
- Normal sampling density
- Normal color

### Scene

- Black / white / gray / custom background
- Orientation axes widget
- XYZ world-coordinate bounds/grid
- Perspective or orthographic camera
- PNG screenshot

### Camera controls

- Fit
- Isometric
- Front / back / left / right / top / bottom
- Zoom in / out
- Orbit left / right / up / down
- Pan left / right / up / down
- Roll left / right
- Focus selected object
- Native mouse trackball interaction

### Director mode

Camera keyframes store:

- time
- name
- camera position
- focal point
- view-up vector
- view angle
- parallel scale / projection

Transition types:

- Smooth
- Orbit
- Linear
- Hold

Director operations:

- Add keyframe
- Update keyframe
- Delete / clear
- Go to shot
- Scrub timeline
- Preview animation
- Save / load camera path JSON
- Export H.264 MP4
- Select FPS
- 1x–4x render-resolution scale

## Planned next

- Mesh sequence / time-lapse OBJ animation track
- Object transform keyframes
- Object visibility / opacity / color keyframes
- Shot cuts and shot library
- Lighting controls and light keyframes
- Clip planes / reveal animations
- Labels / text / scale bar
- Render presets
- Camera-motion presets (push-in, pull-out, arc, fly-by, turntable)
