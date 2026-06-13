import csv
import os
import pytest
from unittest.mock import MagicMock

from logic_export import ExportLogic


# --- Helper: build a minimal mock MarkupsFiducialNode ---

def _make_markup_node(label_coords):
    """label_coords: list of (label, x, y) tuples."""
    node = MagicMock()
    node.GetNumberOfControlPoints.return_value = len(label_coords)
    labels = [lc[0] for lc in label_coords]
    xy = [(lc[1], lc[2]) for lc in label_coords]
    node.GetNthControlPointLabel.side_effect = lambda i: labels[i]

    def _get_pos(i, out):
        out[0] = xy[i][0]
        out[1] = xy[i][1]
        out[2] = 0.0

    node.GetNthControlPointPosition.side_effect = _get_pos
    return node


_ALL_SIX = [
    ("L1_ant",    -0.5,  5.0),
    ("L1_post",    0.5,  5.0),
    ("S1_ant",    -0.5,  0.0),
    ("S1_post",    0.5,  0.0),
    ("FH",         0.0, 10.0),
    ("L1_center",  1.0,  5.0),
]


# --- _check_overwrite ---

class TestCheckOverwrite:
    def test_no_files_returns_paths(self, tmp_path):
        logic = ExportLogic()
        npy, json_, nrrd = logic._check_overwrite(str(tmp_path), "K001", overwrite=False)
        assert npy.endswith("K001_image.npy")
        assert json_.endswith("K001_landmarks.json")
        assert nrrd.endswith("K001_volume.nrrd")

    def test_existing_file_no_overwrite_raises(self, tmp_path):
        (tmp_path / "K001_image.npy").touch()
        logic = ExportLogic()
        with pytest.raises(ValueError, match="既に存在する"):
            logic._check_overwrite(str(tmp_path), "K001", overwrite=False)

    def test_existing_file_with_overwrite_returns_paths(self, tmp_path):
        (tmp_path / "K001_image.npy").touch()
        logic = ExportLogic()
        npy, _, _ = logic._check_overwrite(str(tmp_path), "K001", overwrite=True)
        assert npy.endswith("K001_image.npy")

    def test_multiple_existing_files_listed_in_error(self, tmp_path):
        (tmp_path / "K002_image.npy").touch()
        (tmp_path / "K002_landmarks.json").touch()
        logic = ExportLogic()
        with pytest.raises(ValueError) as exc:
            logic._check_overwrite(str(tmp_path), "K002", overwrite=False)
        msg = str(exc.value)
        assert "K002_image.npy" in msg
        assert "K002_landmarks.json" in msg

    def test_nrrd_only_also_raises(self, tmp_path):
        (tmp_path / "K003_volume.nrrd").touch()
        logic = ExportLogic()
        with pytest.raises(ValueError):
            logic._check_overwrite(str(tmp_path), "K003", overwrite=False)

    def test_different_case_id_does_not_raise(self, tmp_path):
        (tmp_path / "K001_image.npy").touch()
        logic = ExportLogic()
        # K002 has no files → should not raise
        logic._check_overwrite(str(tmp_path), "K002", overwrite=False)


# --- _validate_count ---

class TestValidateCount:
    def test_all_labels_present_does_not_raise(self):
        logic = ExportLogic()
        node = _make_markup_node(_ALL_SIX)
        logic._validate_count(node)

    def test_missing_label_raises(self):
        logic = ExportLogic()
        node = _make_markup_node(_ALL_SIX[:3])  # only 3 labels
        with pytest.raises(ValueError):
            logic._validate_count(node)

    def test_zero_points_raises(self):
        logic = ExportLogic()
        node = _make_markup_node([])
        with pytest.raises(ValueError):
            logic._validate_count(node)

    def test_error_mentions_missing_label(self):
        logic = ExportLogic()
        node = _make_markup_node(_ALL_SIX[:4])  # FH and L1_center missing
        with pytest.raises(ValueError, match="FH"):
            logic._validate_count(node)

    def test_extra_unlabeled_points_still_passes(self):
        # Validate that extra stale points don't break export if required labels are present
        extra = _ALL_SIX + [("unknown_extra", 0.0, 0.0)]
        logic = ExportLogic()
        node = _make_markup_node(extra)
        logic._validate_count(node)  # should not raise


# --- _label_to_index ---

class TestLabelToIndex:
    def test_returns_correct_mapping(self):
        logic = ExportLogic()
        node = _make_markup_node(_ALL_SIX)
        result = logic._label_to_index(node)
        for i, (label, _, _) in enumerate(_ALL_SIX):
            assert result[label] == i

    def test_missing_label_raises(self):
        logic = ExportLogic()
        node = _make_markup_node(_ALL_SIX[:5])  # missing L1_center
        with pytest.raises(ValueError, match="L1_center"):
            logic._label_to_index(node)

    def test_returns_all_six_keys(self):
        logic = ExportLogic()
        node = _make_markup_node(_ALL_SIX)
        result = logic._label_to_index(node)
        assert set(result.keys()) == {"L1_ant", "L1_post", "S1_ant", "S1_post", "FH", "L1_center"}


# --- _collect_available_landmarks_ras_2d ---

class TestCollectAvailableLandmarks:
    def test_collects_all_six_labels(self):
        logic = ExportLogic()
        node = _make_markup_node(_ALL_SIX)
        pts = logic._collect_available_landmarks_ras_2d(node)
        assert set(pts.keys()) == {"L1_ant", "L1_post", "S1_ant", "S1_post", "FH", "L1_center"}

    def test_coordinates_are_correct(self):
        logic = ExportLogic()
        node = _make_markup_node([("FH", 3.0, 7.0)])
        pts = logic._collect_available_landmarks_ras_2d(node)
        assert pts["FH"] == (3.0, 7.0)

    def test_flip_x_axis_negates_x(self):
        logic = ExportLogic(flip_x_axis=True)
        node = _make_markup_node([("FH", 3.0, 7.0)])
        pts = logic._collect_available_landmarks_ras_2d(node)
        assert pts["FH"] == (-3.0, 7.0)

    def test_flip_x_does_not_affect_y(self):
        logic = ExportLogic(flip_x_axis=True)
        node = _make_markup_node([("FH", 3.0, 7.0)])
        pts = logic._collect_available_landmarks_ras_2d(node)
        assert pts["FH"][1] == 7.0

    def test_unknown_labels_are_ignored(self):
        logic = ExportLogic()
        node = _make_markup_node([("UNKNOWN", 0.0, 0.0), ("FH", 1.0, 2.0)])
        pts = logic._collect_available_landmarks_ras_2d(node)
        assert "UNKNOWN" not in pts
        assert "FH" in pts

    def test_partial_set_is_collected(self):
        logic = ExportLogic()
        node = _make_markup_node([("L1_ant", 0.0, 0.0), ("S1_ant", 1.0, 0.0)])
        pts = logic._collect_available_landmarks_ras_2d(node)
        assert len(pts) == 2

    def test_no_flip_x_axis_default(self):
        logic = ExportLogic()
        node = _make_markup_node([("FH", -3.0, 7.0)])
        pts = logic._collect_available_landmarks_ras_2d(node)
        assert pts["FH"][0] == -3.0


# --- export_angles_csv ---

class TestExportAnglesCsv:
    def test_creates_csv_file(self, tmp_path):
        logic = ExportLogic()
        node = _make_markup_node(_ALL_SIX)
        path = logic.export_angles_csv(node, str(tmp_path), "K001")
        assert os.path.exists(path)
        assert path.endswith("angles.csv")

    def test_csv_has_exactly_one_row(self, tmp_path):
        logic = ExportLogic()
        node = _make_markup_node(_ALL_SIX)
        logic.export_angles_csv(node, str(tmp_path), "K001")
        with open(tmp_path / "angles.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_csv_row_has_correct_case_id(self, tmp_path):
        logic = ExportLogic()
        node = _make_markup_node(_ALL_SIX)
        logic.export_angles_csv(node, str(tmp_path), "K042")
        with open(tmp_path / "angles.csv") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["case_id"] == "K042"

    def test_csv_has_angle_columns(self, tmp_path):
        logic = ExportLogic()
        node = _make_markup_node(_ALL_SIX)
        logic.export_angles_csv(node, str(tmp_path), "K001")
        with open(tmp_path / "angles.csv") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
        for angle in ["PI", "PT", "SS", "LL"]:
            assert angle in fieldnames

    def test_csv_base_angle_values_not_empty(self, tmp_path):
        logic = ExportLogic()
        node = _make_markup_node(_ALL_SIX)
        logic.export_angles_csv(node, str(tmp_path), "K001")
        with open(tmp_path / "angles.csv") as f:
            row = next(csv.DictReader(f))
        for angle in ["PI", "PT", "SS", "LL"]:
            assert row[angle] != "", f"{angle} should not be empty"

    def test_csv_angle_values_are_numeric(self, tmp_path):
        logic = ExportLogic()
        node = _make_markup_node(_ALL_SIX)
        logic.export_angles_csv(node, str(tmp_path), "K001")
        with open(tmp_path / "angles.csv") as f:
            row = next(csv.DictReader(f))
        for angle in ["PI", "PT", "SS", "LL"]:
            float(row[angle])  # should not raise

    def test_append_mode_adds_second_row(self, tmp_path):
        logic = ExportLogic()
        node = _make_markup_node(_ALL_SIX)
        logic.export_angles_csv(node, str(tmp_path), "K001")
        logic.export_angles_csv(node, str(tmp_path), "K002")
        with open(tmp_path / "angles.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert {r["case_id"] for r in rows} == {"K001", "K002"}

    def test_overwrite_replaces_existing_row(self, tmp_path):
        logic = ExportLogic()
        node = _make_markup_node(_ALL_SIX)
        logic.export_angles_csv(node, str(tmp_path), "K001")
        logic.export_angles_csv(node, str(tmp_path), "K001", overwrite=True)
        with open(tmp_path / "angles.csv") as f:
            rows = list(csv.DictReader(f))
        k001_rows = [r for r in rows if r["case_id"] == "K001"]
        assert len(k001_rows) == 1

    def test_overwrite_preserves_other_rows(self, tmp_path):
        logic = ExportLogic()
        node = _make_markup_node(_ALL_SIX)
        logic.export_angles_csv(node, str(tmp_path), "K001")
        logic.export_angles_csv(node, str(tmp_path), "K002")
        logic.export_angles_csv(node, str(tmp_path), "K001", overwrite=True)
        with open(tmp_path / "angles.csv") as f:
            rows = list(csv.DictReader(f))
        case_ids = {r["case_id"] for r in rows}
        assert "K001" in case_ids
        assert "K002" in case_ids

    def test_l1pa_present_when_l1_center_placed(self, tmp_path):
        logic = ExportLogic()
        node = _make_markup_node(_ALL_SIX)  # includes L1_center
        logic.export_angles_csv(node, str(tmp_path), "K001")
        with open(tmp_path / "angles.csv") as f:
            row = next(csv.DictReader(f))
        assert row.get("L1PA", "") != ""

    def test_creates_output_dir_if_needed(self, tmp_path):
        logic = ExportLogic()
        node = _make_markup_node(_ALL_SIX)
        new_dir = str(tmp_path / "subdir" / "deep")
        logic.export_angles_csv(node, new_dir, "K001")
        assert os.path.exists(new_dir)
