Third-party components and sources
==================================

LinkODT uses the following components for its interface, computation, and visualization. See :doc:`requirements` for dependencies and versions.

.. list-table:: Main components
   :header-rows: 1

   * - Component
     - Purpose
   * - PySide6 / Qt
     - Graphical interface
   * - PyTorch, timm, DINOv2
     - Segmentation models and feature encoders
   * - NumPy, SciPy, scikit-image, pandas
     - Image processing, numerical computation, and result tables
   * - NetworkX
     - Network analysis
   * - VTK, PyVista, Open3D
     - Three-dimensional geometry and visualization
   * - imageio-ffmpeg, Pillow
     - Image and movie export

Third-party components retain their own licenses and source notices. The Windows package includes license files in ``_internal/THIRD_PARTY_LICENSES/``. Cite the original papers for external models and analysis methods used in published results; see :doc:`citing`.
