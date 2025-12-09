# Sagittal Measure Assist (3D Slicer Extension)

Lateral spine X-ray helper for sagittal alignment measurement and ML data creation.

## Modules
- `SagittalMeasureAssist`: Manual workflow to place 5 landmarks (L1_ant, L1_post, S1_ant, S1_post, FH) and compute PI/PT/SS/LL. Later will host model loading + auto-proposal buttons.
- `SagittalMeasureTrain`: Data export helper to build a training set for keypoint models (image arrays + landmark JSON in IJK/pixel space).

## Current workflow
1) Load lateral X-ray DICOM as a volume.  
2) In **SagittalMeasureTrain**, create/select Markups and place 5 landmarks in order.  
3) Export a sample: saves `.npy` (image array), `.nrrd` (source volume), `.json` (landmarks in IJK with spacing/origin).  
4) Train a keypoint model externally (PyTorch/ONNX) using the exported dataset.  
5) Bring the trained model back to Slicer (planned) to auto-place landmarks and reuse the existing angle computation.

## Directory structure
- `SagittalMeasureAssist/` — Manual measurement module (angles, UI, logic).  
- `SagittalMeasureTrain/` — Training-data export module.  
- `CMakeLists.txt` — Extension entry point.

## Roadmap
- Add model loader + auto-plot button to `SagittalMeasureAssist` (likely via ONNX Runtime).  
- Provide a reference training script (PyTorch) for the exported format.  
- Add validation/overlay previews and basic QA checks on export.
