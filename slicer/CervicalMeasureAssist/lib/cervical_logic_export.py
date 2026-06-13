"""
Export helpers for cervical landmark training data and angle CSV output.
"""

import csv
import json
import os

import numpy as np
import slicer
import vtk

from logic_angles_cervical import compute_cervical_measurements
from cervical_measurement_sets import CERVICAL_SET, MeasurementSetDef


class ExportLogic:
    def __init__(self, flip_x_axis=False, measurement_set: MeasurementSetDef = None):
        self.flip_x_axis = flip_x_axis
        self._set = measurement_set if measurement_set is not None else CERVICAL_SET

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

    def _label_to_index(self, markupNode):
        labels = self._set.point_labels
        mapping = {}
        for i in range(markupNode.GetNumberOfControlPoints()):
            label = markupNode.GetNthControlPointLabel(i)
            if label in labels:
                mapping[label] = i
        missing = [l for l in labels if l not in mapping]
        if missing:
            raise ValueError(f"ラベルが見つかりません: {', '.join(missing)}")
        return mapping

    def _collect_landmarks_ijk(self, markupNode, volumeNode):
        label_idx = self._label_to_index(markupNode)
        coords = {}
        ras = [0.0, 0.0, 0.0]
        for label in self._set.point_labels:
            markupNode.GetNthControlPointPositionWorld(label_idx[label], ras)
            ijk = self._ras_to_ijk(volumeNode, ras)
            coords[label] = {"i": float(ijk[0]), "j": float(ijk[1]), "k": float(ijk[2])}
        return coords

    def _collect_landmarks_ras_2d(self, markupNode):
        label_idx = self._label_to_index(markupNode)
        points = {}
        coordsRAS = [0.0, 0.0, 0.0]
        for label in self._set.point_labels:
            markupNode.GetNthControlPointPosition(label_idx[label], coordsRAS)
            x = -coordsRAS[0] if self.flip_x_axis else coordsRAS[0]
            points[label] = (x, coordsRAS[1])
        return points

    def _volume_metadata(self, volumeNode):
        spacing = volumeNode.GetSpacing()
        ijk_to_ras = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijk_to_ras)
        direction = [[ijk_to_ras.GetElement(r, c) for c in range(3)] for r in range(3)]
        origin = [ijk_to_ras.GetElement(r, 3) for r in range(3)]
        return {"spacing": list(spacing), "ijk_to_ras": direction, "origin_ras": origin}

    def _validate_count(self, markupNode):
        placed = {markupNode.GetNthControlPointLabel(i) for i in range(markupNode.GetNumberOfControlPoints())}
        missing = [lbl for lbl in self._set.point_labels if lbl not in placed]
        if missing:
            raise ValueError(f"ラベルが見つかりません: {', '.join(missing)}")

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
        angles_deg = self._set.compute_fn(points_ras_2d)

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

    def export_angles_csv(self, markupNode, outputDir, caseId, overwrite=False):
        os.makedirs(outputDir, exist_ok=True)
        points = self._collect_available_landmarks_ras_2d(markupNode)
        angles = self._set.compute_fn(points)

        csv_path = os.path.join(outputDir, "angles.csv")
        fieldnames = ["case_id"] + self._set.angle_names

        row = {"case_id": caseId}
        for name in self._set.angle_names:
            val = angles.get(name)
            row[name] = f"{val:.2f}" if isinstance(val, float) else ""

        if overwrite and os.path.exists(csv_path):
            rows = []
            found = False
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("case_id") == caseId:
                        rows.append(row)
                        found = True
                    else:
                        rows.append(r)
            if not found:
                rows.append(row)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
        else:
            file_exists = os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

        return csv_path

    def _collect_available_landmarks_ras_2d(self, markupNode):
        labels_set = set(self._set.point_labels)
        points = {}
        coordsRAS = [0.0, 0.0, 0.0]
        for i in range(markupNode.GetNumberOfControlPoints()):
            label = markupNode.GetNthControlPointLabel(i)
            if label in labels_set:
                markupNode.GetNthControlPointPosition(i, coordsRAS)
                x = -coordsRAS[0] if self.flip_x_axis else coordsRAS[0]
                points[label] = (x, coordsRAS[1])
        return points
