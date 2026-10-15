System requirements and dependencies
========================================================================

Software
----------------

The standalone Windows package runs without a separate Python installation. For the source edition, use Python 3.11 and install the project dependencies; the Studio runtime modules require this Python version.

The module 4 backend supports both NumPy 1.x and 2.x integration APIs. For dependency compatibility issues, see :doc:`faq`.

Hardware planning
----------------------------------

The project has no benchmark-validated minimum RAM, GPU-memory, or CPU specification. Assess hardware using the workload:

* For image browsing and small batches, first test TIFF loading and GUI responsiveness.
* For large volumes, voxel sets, surface meshes, skeletons, and adjacency structures may occupy memory simultaneously.
* For training/prediction, GPU-memory requirements vary with backbone, tile size, batch size, and numerical precision.
* For animation export, higher resolution increases rendering time, GPU memory, and storage requirements.

For example, a 100×1024×1024 uint16 stack uses about 200 MiB for the raw array alone. Floating-point intermediates, labels, and geometry require additional memory. This illustrates input size rather than specifying a minimum machine configuration.

When resources are insufficient
--------------------------------------------------------------

Reduce batch size and the number of simultaneously loaded samples. For training, reduce **Batch** or select a smaller suitable model. Tile size is an internal source setting, with no corresponding training GUI control. Estimate peak memory with a representative small volume before planning a full batch. Preview video at 1× resolution before higher-resolution export.

Models and network access
--------------------------------------------------

The current project copy does not include every pretrained model file. Download scripts require access to the relevant model sources. On offline machines, prepare weights in advance and place them according to configuration. GGUF models are used only by the optional AI Director; manual three-dimensional presentation works without a language model.
