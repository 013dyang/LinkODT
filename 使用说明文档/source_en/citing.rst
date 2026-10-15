Usage records and citation
====================================================

Record the analysis version
------------------------------------------------------

In papers, reports, and shared datasets, record the project name, version/date, model architecture, weight source, key parameters, and calibration. Folder names and ``VERSION.txt`` help identify the version. Retain the corresponding source version and environment information with the analysis results.

A formal LinkODT publication DOI is not supplied with this version. When a publication or archived release becomes available, use the standard citation provided by the maintainers.

Methods-record template
----------------------------------------------

Fill in the actual values in this structure:

   LinkODT [version/commit] was used to analyze [imaging modality and objects]. Foreground masks were generated with [segmentation architecture, backbone, and weights] and checked through [review workflow]. Instance analysis used [Physical/Social level], with spatial calibration [X, Y, Z]. Networks were constructed using [edge definitions and thresholds]. The time interval was [Δt], and the tracking search radius was [R]. Results comprised [sample count and biological replicates], with [statistical method].

When using DINOv2, U-Net, SegFormer, or other external methods, cite their original papers and implementation versions.
