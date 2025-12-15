"""
Export helpers for sagittal landmark training data.
Keeps I/O and coordinate transforms out of the main widget.
"""

import json
import os

import numpy as np
import slicer
import vtk

from logic_angles import compute_angles_from_points

REQUIRED_LABELS_ORDERED = ["L1_ant", "L1_post", "S1_ant", "S1_post", "FH"]


class ExportLogic:
    def __init__(self, flip_x_axis=False):
        self.flip_x_axis = flip_x_axis

    def _check_overwrite(self, outputDir, caseId, overwrite):
        npy_path = os.path.join(outputDir, f"{caseId}_image.npy")
        json_path = os.path.join(outputDir, f"{caseId}_landmarks.json")
        nrrd_path = os.path.join(outputDir, f"{caseId}_volume.nrrd")
        exists = [p for p in (npy_path, json_path, nrrd_path) if os.path.exists(p)]
        if exists and not overwrite:
            raise ValueError(f"既に存在するファイルがあります: {', '.join(os.path.basename(p) for p in exists)}")
        return npy_path, json_path, nrrd_path

    def _ras_to_ijk(self, volumeNode, ras_point):
        ras_to_ijk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(ras_to_ijk)
        ras_h = [ras_point[0], ras_point[1], ras_point[2], 1.0]
        ijk_h = ras_to_ijk.MultiplyPoint(ras_h)
        return ijk_h[:3]

    def _collect_landmarks_ijk(self, markupNode, volumeNode):
        self._validate_count(markupNode)
        coords = {}
        ras = [0.0, 0.0, 0.0]
        for i, label in enumerate(REQUIRED_LABELS_ORDERED):
            markupNode.SetNthControlPointLabel(i, label)
            markupNode.GetNthControlPointPositionWorld(i, ras)
            ijk = self._ras_to_ijk(volumeNode, ras)
            coords[label] = {"i": float(ijk[0]), "j": float(ijk[1]), "k": float(ijk[2])}
        return coords

    def _collect_landmarks_ras_2d(self, markupNode):
        self._validate_count(markupNode)
        points = {}
        coordsRAS = [0.0, 0.0, 0.0]
        for idx, label in enumerate(REQUIRED_LABELS_ORDERED):
            markupNode.SetNthControlPointLabel(idx, label)
            markupNode.GetNthFiducialPosition(idx, coordsRAS)
            x = -coordsRAS[0] if self.flip_x_axis else coordsRAS[0]
            y = coordsRAS[1]
            points[label] = (x, y)
        return points

    def _volume_metadata(self, volumeNode):
        spacing = volumeNode.GetSpacing()
        ijk_to_ras = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijk_to_ras)
        direction = [[ijk_to_ras.GetElement(r, c) for c in range(3)] for r in range(3)]
        origin = [ijk_to_ras.GetElement(r, 3) for r in range(3)]
        return {"spacing": list(spacing), "ijk_to_ras": direction, "origin_ras": origin}

    def _validate_count(self, markupNode):
        if markupNode.GetNumberOfControlPoints() != len(REQUIRED_LABELS_ORDERED):
            raise ValueError("マークアップ点が5個ではありません。指定の順番で5点を配置してください。")

    def export_training_sample(self, volumeNode, markupNode, outputDir, caseId, overwrite=False):
        self._validate_count(markupNode)
        os.makedirs(outputDir, exist_ok=True)
        npy_path, json_path, nrrd_path = self._check_overwrite(outputDir, caseId, overwrite)

        image_array = slicer.util.arrayFromVolume(volumeNode)
        np.save(npy_path, image_array)

        slicer.util.saveNode(volumeNode, nrrd_path)

        landmarks_ijk = self._collect_landmarks_ijk(markupNode, volumeNode)
        metadata = self._volume_metadata(volumeNode)
        points_ras_2d = self._collect_landmarks_ras_2d(markupNode)
        angles_deg = compute_angles_from_points(points_ras_2d)

        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "case_id": caseId,
                    "landmarks_ijk": landmarks_ijk,
                    "metadata": metadata,
                    "image_shape": list(image_array.shape),
                    "angles_deg": angles_deg,
                    "flip_x_axis": bool(self.flip_x_axis),
                },
                fp,
                ensure_ascii=False,
                indent=2,
            )

        return {"npy": npy_path, "json": json_path, "nrrd": nrrd_path}
