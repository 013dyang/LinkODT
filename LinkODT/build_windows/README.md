# Windows EXE build

This preserves the previously used PyInstaller **ONEDIR** strategy.

Why ONEDIR:
- PyTorch
- DINOv2
- VTK / PyVista
- Open3D
- Qt

make a one-file executable unnecessarily large and fragile.

## Build

1. Confirm `LinkODT.py` runs normally from Python.
2. Put required weights in `models/`.
3. Double-click `BUILD_LinkODT_EXE.bat`.
4. Final executable:
   `dist/LinkODT/LinkODT.exe`
5. Give users the **entire `dist/LinkODT` folder**, not the EXE alone.

The builder copies:
- `models/`
- `dinov2-main/`

into the PyInstaller `_internal` runtime folder after the build.
