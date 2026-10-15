Frequently asked questions and troubleshooting
============================================================================================

RUN_LinkODT.bat does not start the application
--------------------------------------------------------------------------------------------

Use ``RUN_LinkODT.bat`` in the LinkODT project directory. It selects ``LINKODT_PYTHON`` when supplied, the local ``.linkodt_runtime`` when present, an activated Conda environment, or Python on PATH. Confirm that the selected environment contains the project dependencies. You can also activate it and run ``python LinkODT.py``. The old ``RUN_LinkNex.bat`` remains a launcher alias.

Running main.py does not open LinkODT
--------------------------------------------------------------------------

The root-level ``main.py`` is an example. The main entry point is ``LinkODT.py``; the independent visualization entry point is ``visualization/main.py``.

Missing pyvistaqt, Qt, or VTK
----------------------------------------------------------

Install dependencies into the environment actually used to launch LinkODT, then rerun ``tools/check_environment.py``. For QtCore DLL import errors, check Python/Qt compatibility and installation integrity.

Network calculation reports that numpy has no trapezoid
--------------------------------------------------------------------------------------------------------------

LinkODT 3.2 uses ``numpy.trapezoid`` when available and the equivalent ``numpy.trapz`` under NumPy 1.x. If this error still appears, check that you launched the current project copy. Check the final log and output tables after rerunning.

DINOv2 or segmentation weights cannot be found
--------------------------------------------------------------------------------------------

Check architecture, backbone, and weight compatibility. Pretrained encoder weights omit the task decoder. Inspect ``models/``, download scripts, and ``MISSING_EXTERNAL_FILES.txt``. Training/prediction logs identify loading failures.

Module 3 cannot process a single slice
----------------------------------------------------------------------------

The full instance pipeline requires at least two Z slices. Inspect single-slice masks in the two-dimensional segmentation interface. Three-dimensional measurements require an actual Z stack; copying one slice does not establish real object thickness.

Existing results are not detected
------------------------------------------------------------------

Check matching sample stems and open the immediate parent or ``processed_output``. Modules 3/4 search expected locations rather than arbitrary nested folders. A standalone OBJ does not contain all inputs required to reconstruct a network.

Network parameter changes leave results unchanged
--------------------------------------------------------------------------------------------------

Check whether existing outputs were reused. Select Overwrite existing network results, rerun, and verify timestamps and settings. Changing a field alone leaves saved networks unchanged.

The dynamics viewer becomes black after mask generation
--------------------------------------------------------------------------------------------------------------

LinkODT 3.2 routes worker completion and error callbacks through GUI-thread QObject slots. Generation of eight demo masks, automatic entry into mask view, and subsequent frame navigation passed without Qt thread warnings. For an unexpected empty view, check the selected dataset/frame, mask foreground, and log. Include your input dimensions and version when reporting the problem.

Mask review has no brush controls
------------------------------------------------------------------

The current GUI has no connected add/erase controls. Use it for viewing and mask generation. Manual revisions require a separate TIFF editing workflow, with synchronized frame masks and display caches; see :doc:`dynamics`.

Three-dimensional measurements do not look like µm
----------------------------------------------------------------------------------------------------

Module 3 morphology uses unit voxel calibration. Module 4 XYZ scale affects network coordinates/distances. Dynamics uses its own XY, Z, and Δt settings. Check unit provenance separately in each module.

Can image-intensity tables be compared directly?
------------------------------------------------------------------------------------------------

Confirm that instance analysis read the matching original image and that acquisition, background correction, and intensity processing are consistent. Missing original images lead to mask-derived placeholder intensities. Display brightness and pseudocolors are separate from raw fluorescence intensity.

What should a problem report include?
--------------------------------------------------------------------------

Record the module, button sequence, input dimensions/axis order, parameters, version, and full error. Provide a shareable small example and log, explaining expected and actual behavior. Include the selected sample and settings in screenshots.
