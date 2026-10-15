# LinkODT

Organelle imaging, segmentation, network analysis and 3D visualization.

Developed by Dan LI and Dong YANG · Dayong Jin’s ODT Group.

## Run the application

For source development, use Python 3.11. The application is in `LinkODT/`.

```powershell
cd LinkODT
conda env create -f environment.yml
conda activate linkodt
python LinkODT.py
```

Review the environment file before creating an environment. For an existing
working PyTorch installation, keep its compatible torch/torchvision versions
and install missing project dependencies as described in the user guide.
`RUN_LinkODT.bat` also starts the application with an existing environment.

The Windows standalone package includes its own runtime. Download that package
from a published GitHub Release when one is available, then extract the whole
folder and run `LinkODT.exe`. Keep `_internal/` and `UserGuide/` beside it.

## User guide

Chinese and English guide sources and screenshots are in `使用说明文档/`.
Build them from the repository root:

```powershell
python -m pip install -r "使用说明文档/requirements-docs.txt"
python -m sphinx -b html "使用说明文档/source" "使用说明文档/html"
python -m sphinx -b html "使用说明文档/source_en" "使用说明文档/en/html"
```

Open `使用说明文档/index.html` after building. Small synthetic practice data are
included in `使用说明文档/demo_data/`. Real microscopy data, model checkpoints,
Python environments, build outputs and local maintenance records stay outside
version control. Screenshots in the guide illustrate the real-data workflow;
the underlying real datasets are supplied separately.

## Studio implementation

The editable Studio UI and display-window controls are in
`LinkODT/visualization/studio_ui.py` and `display_controls.py`.
Three CPython 3.11 modules supplied with Dan LI’s Studio release are required:
`_studio_core.pyc`, `_studio_io.pyc`, and `_studio_director.pyc`.
They are intentionally tracked. Obtain the matching original Python sources
before revising their underlying rendering and TIFF-reading implementation.
See `LinkODT/visualization/README_IMPLEMENTATION.md` and `studio_origin.json`.

## Collaboration

Keep `main` runnable. Make changes on a feature branch and submit a pull request.
Update both guide languages when visible controls or procedures change.
See [CONTRIBUTING.md](CONTRIBUTING.md).

Third-party components retain their own licenses. LinkODT’s authors have not
selected a public open-source license for this repository.
