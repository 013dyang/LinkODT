Studio alignment, 14 October 2026

The Studio executable supplied by Dan LI on 22 September contains compiled
CPython 3.11 modules, without editable Python source. The three modules retained
here preserve the supplied TIFF readers, camera interpolation, volume renderer,
time playback and scene transitions. studio_origin.json records their origin.

studio_ui.py provides editable LinkODT menus, scrolling panels, shared themes,
typography and navigation. No external LinkNex_Studio_v3 folder is required.
Use Python 3.11 for source execution; the packaged Windows build carries its
own interpreter. When original source becomes available, replace the retained
compiled modules after comparing behavior with the alignment checks.
