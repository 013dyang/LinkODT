Setting up Python
==================================

Set up the Python environment before starting LinkODT. The graphical interface uses PySide6 and VTK; deep-learning training and prediction use PyTorch.

1. Locate the project directory
--------------------------------------------------------------

Open the folder containing ``LinkODT.py``, ``semantic_models.py``, ``punctate_dynamics_module.py``, and ``visualization/``. The outer project-copy folder is a container. Run subsequent project commands from the actual source root.

2. Create an environment
------------------------------------------------

If Conda is available, create a dedicated environment using the supplied file:

.. code-block:: powershell

   conda env create -f environment.yml
   conda activate linkodt

``environment.yml`` specifies Python 3.11 and the basic dependencies. Select a compatible PyTorch / torchvision installation for your GPU and driver using the `official PyTorch installation selector <https://pytorch.org/get-started/locally/>`_.

If you already have a working PyTorch environment, install the project dependencies in that environment:

.. code-block:: powershell

   python -m pip install -r requirements_linknex.txt

3. Check the environment
------------------------------------------------

.. code-block:: powershell

   python tools/check_environment.py
   python tools/project_self_check.py

Verify that the Python path points to the intended environment. Check imports for numpy, scipy, PySide6, vtk, pyvista, pyvistaqt, torch, and other dependencies. When GPU execution is required, confirm that ``CUDA available`` is true. The environment check also identifies missing DINOv2 weights.

4. Install optional features
--------------------------------------------------------

For SegFormer-B2:

.. code-block:: powershell

   python -m pip install -r requirements_semantic_optional.txt

Basic Visualization functionality requires PySide6, VTK, numpy, imageio, and imageio-ffmpeg. The local Qwen AI Director additionally requires ``llama-cpp-python`` and a compatible GGUF model. See ``visualization/INSTALL_LOCAL_QWEN.md`` for configuration details.

5. Environment used for this guide
--------------------------------------------------------------------

Screenshots were captured on Windows with PySide6 6.8.3. Reading this HTML guide requires only a browser. To run LinkODT, configure your own environment using the preceding steps and check its dependencies before analysis.
