# Backend source snapshots

These files are kept as readable algorithm/reference sources.

Important: the current `LinkNex.py` release embeds the active Module 3,
Module 4 and point-cloud-editor backend code internally so the GUI can be
distributed as one main Python application.

Therefore editing a file in `backend_sources/` does **not automatically**
change the currently embedded GUI backend.

Use this folder as:
- readable algorithm source;
- development/reference code;
- a future basis for modular refactoring.

The current GUI remains the runtime source of truth until the embedded backend
is deliberately replaced with normal Python imports.
