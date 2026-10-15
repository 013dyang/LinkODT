LinkODT Visualization Studio — v0.2
===================================

Module 6 is a standalone 3D visualization and camera-animation application.
The main LinkODT home screen launches this software in a separate process.

Start
-----
From LinkODT:
    click Visualization

Or directly:
    visualization/RUN_Visualization.bat
    visualization/main.py

Current inputs
--------------
1. OBJ mesh (.obj)
2. LinkODT / generic point-cloud TXT or CSV

TXT support includes common layouts such as:
    x,y,z
    x,y,z,nx,ny,nz
    x,y,z,r,g,b
    x,y,z,nx,ny,nz,r,g,b
    x,y,z,r,g,b,nx,ny,nz
and optional trailing ID/scalar columns.
Header-based x/y/z, r/g/b and nx/ny/nz formats are also recognized.

3D display controls
-------------------
- Multiple mesh / point-cloud objects in one scene
- Per-object visibility
- Object color or file RGB
- Surface / surface+edges / wireframe / points display
- Point size
- Line width
- Opacity
- Point/surface normals with adjustable length, density and color
- Background: black / white / gray / custom
- Orientation axes
- XYZ world-coordinate bounds/grid
- Perspective / orthographic projection
- Camera presets: fit, iso, front, back, left, right, top, bottom
- Camera actions: zoom, orbit, pan, roll, focus selected
- PNG screenshot export

Director Mode
-------------
The movie system is camera-keyframe based.

Typical workflow:
    1. Arrange the first camera view with mouse / camera buttons.
    2. Set time = 0 s and click Add Keyframe.
    3. Move / rotate / zoom / pan to the next shot.
    4. Set a later time and Add Keyframe.
    5. Choose how the camera enters that shot:
         Smooth  - smooth point-to-point camera movement
         Orbit   - curved camera orbit around the focal point
         Linear  - direct linear movement
         Hold    - hold the previous shot until this keyframe
    6. Preview the complete timeline.
    7. Export MP4.

Camera paths can be saved to / loaded from JSON.

3D ROI / Focus
--------------
- Pick focus points directly on OBJ mesh / point cloud
- Visible spherical ROI markers
- Rename and resize ROIs
- Focus / Frame ROI
- Save / load ROI JSON

AI Director (Local Qwen GGUF)
------------------------------
- Natural-language shot description in Chinese or English
- Fully local GGUF inference through llama-cpp-python
- Choose your own Qwen3-4B / 8B / 14B Q4_K_M .gguf
- No API key / Base URL / cloud request
- Restricted validated action vocabulary
- Rich camera actions: orbit, turntable, push/pull, truck, pedestal, dolly, look/tilt, roll, ROI/object focus, lens/projection and timing
- Smooth / linear / orbit / ease / hold / cut transitions
- Review generated plan before applying it
- Large-angle orbit/turntable is automatically split into real intermediate camera keyframes
- See LOCAL_AI_DIRECTOR_V0_3.md and INSTALL_LOCAL_QWEN.md

MP4 export
----------
- User-selectable FPS
- 1x to 4x render-resolution scale
- H.264 MP4 via imageio-ffmpeg

Planned next-stage features
---------------------------
- Mesh/time-sequence animation tracks
- Object appearance keyframes
- Clip planes / reveal effects
- Scene light controls
- Multiple camera cuts / shot library
- Text labels / annotations
- Depth-of-field style post-processing where practical
- Preset cinematic camera moves
- Render presets for publication / presentation / social video

Dependencies
------------
Install:
    pip install -r requirements_visualization.txt

The main LinkODT environment already lists most of these packages.
