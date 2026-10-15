Workstations and remote execution
==================================================================

Execution workflow
------------------------------------

LinkODT is a desktop GUI project. The current version has no verified one-click Slurm / HPC batch entry point. For remote GUI operation, prepare a graphical desktop and dependencies on the workstation, connect through remote desktop, and launch ``LinkODT.py`` there.

1. Copy the project, data, and models into dedicated workstation directories.
2. Configure the environment on the workstation and run the checks. The local Python environment is separate from the remote environment.
3. Confirm the remote GPU, driver, and CUDA before training or prediction.
4. Test TIFF loading, VTK rendering, and output permissions with a small sample.
5. Bring back tables, logs, parameters, and presentation files while retaining sample-directory relationships.

Separate computation and inspection
----------------------------------------------------------------------

Generate instance results on computer A and inspect them through direct loading in modules 3/4 on computer B. Copy skeletons, point clouds, label mappings, and CSVs together. Network PKL stores intermediate objects and depends on compatible environments. Include CSVs when exchanging quantitative results.

GUI operation requires an appropriate display environment. To integrate backends with a queue on a headless node, a developer should first define an input interface, logs, and exit status, then compare a small sample against the GUI results. This guide does not provide a verified command for that extension.

Check transferred files
----------------------------

Compare relative paths, file counts, and sizes before and after transfer. Open a representative image and its matching mask after transfer; check that both load correctly and retain the expected dimensions. Write long-running jobs to a workspace with sufficient storage and retain logs to distinguish complete, stopped, and partial outputs.
