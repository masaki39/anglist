import math
import pytest

import SagittalMeasureAssist.logic_angles as angles


def test_vector_from_points():
    assert angles.vector_from_points((0, 0), (1, 1)) == (1, 1)


def test_vector_length_zero():
    assert angles.vector_length((0, 0)) == 0


@pytest.mark.parametrize(
    "v,expected",
    [
        ((1, 0), 0.0),
        ((0, 1), -90.0),  # y増加が下向きなので負
        ((0, -1), 90.0),
        ((-1, 0), 0.0),
    ],
)
def test_signed_slope_angle_deg(v, expected):
    assert math.isclose(angles.signed_slope_angle_deg(v), expected, abs_tol=1e-6)


@pytest.mark.parametrize(
    "v,expected",
    [
        ((0, -1), 0.0),
        ((1, 0), 90.0),
        ((-1, 0), -90.0),
        ((0, 1), 180.0 - 180.0),  # wrapped to 0 after branch
    ],
)
def test_signed_vertical_angle_deg(v, expected):
    assert math.isclose(angles.signed_vertical_angle_deg(v), expected, abs_tol=1e-6)


def test_pelvic_incidence_deg_basic():
    v_pelvis = (0, -1)
    v_S1 = (1, 0)
    assert math.isclose(angles.pelvic_incidence_deg(v_pelvis, v_S1), 0.0, abs_tol=1e-6)


def test_compute_angles_from_points():
    pts = {
        "FH": (0.5, 2.0),
        "S1_ant": (0.0, 0.0),
        "S1_post": (1.0, 1.0),
        "L1_ant": (0.0, 1.0),
        "L1_post": (1.0, 2.0),
    }
    result = angles.compute_angles_from_points(pts)
    assert set(result.keys()) == {"PI", "PT", "SS", "LL"}
    assert math.isclose(result["SS"], -45.0, abs_tol=0.5)
    assert math.isclose(result["LL"], 0.0, abs_tol=0.5)
    assert math.isclose(result["PT"], 0.0, abs_tol=0.5)
    assert math.isclose(result["PI"], 45.0, abs_tol=0.5)


def test_compute_angles_missing():
    with pytest.raises(ValueError):
        angles.compute_angles_from_points({"FH": (0, 0)})
