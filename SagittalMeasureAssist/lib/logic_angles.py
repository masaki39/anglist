"""
Utility functions for sagittal parameter angle computations.
Separated from the main module to keep logic readable and ready for future AI integration.
"""

import math


def vector_from_points(a, b):
    """Return vector from point a to b."""
    return (b[0] - a[0], b[1] - a[1])


def vector_length(v):
    """Euclidean length of a 2D vector."""
    return math.hypot(v[0], v[1])


def normalize(v):
    """Return unit vector; raise on zero length."""
    length = vector_length(v)
    if length == 0:
        raise ValueError("Zero-length vector encountered during normalization.")
    return (v[0] / length, v[1] / length)


def angle_between_vectors(v1, v2):
    """
    Compute absolute angle (0-180) between two 2D vectors.
    """
    len1 = vector_length(v1)
    len2 = vector_length(v2)
    if len1 == 0 or len2 == 0:
        raise ValueError("Cannot compute angle with zero-length vector.")
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cos_theta = max(min(dot / (len1 * len2), 1.0), -1.0)
    theta_rad = math.acos(cos_theta)
    return math.degrees(theta_rad)


def signed_slope_angle_deg(v):
    """
    Signed angle to horizontal in [-90, 90]; y increases downward, so sign flipped.
    """
    if vector_length(v) == 0:
        raise ValueError("Cannot compute slope for zero-length vector.")
    ang = math.degrees(math.atan2(v[1], v[0]))  # [-180,180]
    if ang > 90:
        ang -= 180
    elif ang < -90:
        ang += 180
    return -ang


def signed_vertical_angle_deg(v):
    """
    Signed angle to vertical (headward) in [-90, 90]; x>0 (anterior) is positive.
    """
    if vector_length(v) == 0:
        raise ValueError("Cannot compute vertical angle for zero-length vector.")
    ang = math.degrees(math.atan2(v[0], -v[1]))  # [-180,180]
    if ang > 90:
        ang -= 180
    elif ang < -90:
        ang += 180
    return ang


def wrap_signed_angle(angle):
    """Wrap to [-180, 180]."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def pelvic_incidence_deg(v_pelvis, v_S1):
    """
    PI defined as |90 - angle(pelvis vector, S1 line)| -> range [0,90].
    """
    theta = angle_between_vectors(v_pelvis, v_S1)
    return abs(90.0 - theta)


def lumbosacral_lordosis_deg(v_L1, v_S1):
    """
    Signed L1-S1 Cobb: slope(S1) - slope(L1), wrapped to [-180,180].
    """
    slope_L1 = signed_slope_angle_deg(v_L1)
    slope_S1 = signed_slope_angle_deg(v_S1)
    ll = slope_S1 - slope_L1
    return wrap_signed_angle(ll)


REQUIRED_KEYS = ["FH", "S1_ant", "S1_post", "L1_ant", "L1_post"]


def compute_angles_from_points(points):
    """
    Compute PI, PT, SS, and LL from landmark points.

    Args:
        points (dict): Mapping of landmark names to (x, y) tuples.
    Returns:
        dict: {"PI": deg, "PT": deg, "SS": deg, "LL": deg}
    """
    missing = [k for k in REQUIRED_KEYS if k not in points]
    if missing:
        raise ValueError(f"Missing points: {', '.join(missing)}")

    FH = points["FH"]
    S1_ant = points["S1_ant"]
    S1_post = points["S1_post"]
    L1_ant = points["L1_ant"]
    L1_post = points["L1_post"]

    v_S1 = vector_from_points(S1_ant, S1_post)
    v_L1 = vector_from_points(L1_ant, L1_post)
    S1_mid = ((S1_ant[0] + S1_post[0]) / 2.0, (S1_ant[1] + S1_post[1]) / 2.0)
    v_pelvis = vector_from_points(FH, S1_mid)

    SS = signed_slope_angle_deg(v_S1)
    PT = signed_vertical_angle_deg(v_pelvis)
    LL = lumbosacral_lordosis_deg(v_L1, v_S1)
    PI_modified = pelvic_incidence_deg(v_pelvis, v_S1)

    return {"PI": PI_modified, "PT": PT, "SS": SS, "LL": LL}
