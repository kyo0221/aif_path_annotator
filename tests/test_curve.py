import numpy as np
import pytest

from tools.path_annotator import curve


def make_label(first=25, last=47, value=0.5):
    xp = [0.0] * curve.N_ROWS
    h = [0] * curve.N_ROWS
    for i in range(first, last + 1):
        xp[i] = value
        h[i] = 1
    return xp, h


def test_control_rows_spread_over_valid_rows():
    _, h = make_label(25, 47)
    assert curve.control_rows(h, 3) == [25, 36, 47]


def test_control_rows_k1_is_middle_of_valid():
    _, h = make_label(25, 47)
    assert curve.control_rows(h, 1) == [36]


def test_weights_sum_to_one_on_valid_rows():
    _, h = make_label(25, 47)
    w = curve.weights(h, 3)
    assert w.shape == (3, curve.N_ROWS)
    valid = [i for i, v in enumerate(h) if v]
    np.testing.assert_allclose(w[:, valid].sum(axis=0), 1.0, atol=1e-12)


def test_weights_are_zero_on_invalid_rows():
    _, h = make_label(25, 47)
    w = curve.weights(h, 3)
    invalid = [i for i, v in enumerate(h) if not v]
    assert w[:, invalid].max() == 0.0


def test_equal_deltas_translate_whole_curve():
    xp, h = make_label(25, 47, 0.5)
    out = curve.apply_control_delta(xp, h, [0.1, 0.1, 0.1])
    for i in range(25, 48):
        assert out[i] == pytest.approx(0.6, abs=1e-12)


def test_single_control_point_is_pure_shift():
    xp, h = make_label(25, 47, 0.5)
    out = curve.apply_control_delta(xp, h, [-0.2])
    for i in range(25, 48):
        assert out[i] == pytest.approx(0.3, abs=1e-12)


def test_invalid_rows_are_untouched():
    xp, h = make_label(25, 47, 0.5)
    xp[10] = 0.77
    out = curve.apply_control_delta(xp, h, [0.1, 0.2, 0.3])
    assert out[10] == 0.77
    assert out[0] == 0.0


def test_result_is_clipped_to_unit_range():
    xp, h = make_label(25, 47, 0.5)
    hi = curve.apply_control_delta(xp, h, [9.0, 9.0, 9.0])
    lo = curve.apply_control_delta(xp, h, [-9.0, -9.0, -9.0])
    assert max(hi) == 1.0 and min(hi[25:]) == 1.0
    assert min(lo) == 0.0 and max(lo[25:]) == 0.0


def test_local_drag_moves_near_rows_more_than_far_rows():
    xp, h = make_label(25, 47, 0.5)
    out = curve.apply_control_delta(xp, h, [0.0, 0.2, 0.0])
    near = abs(out[36] - 0.5)
    far = abs(out[47] - 0.5)
    assert near > far


def test_apply_shift_matches_single_control_point():
    xp, h = make_label(25, 47, 0.4)
    assert curve.apply_shift(xp, h, 0.05) == curve.apply_control_delta(xp, h, [0.05])


def test_no_valid_rows_returns_input_unchanged():
    xp = [0.3] * curve.N_ROWS
    h = [0] * curve.N_ROWS
    assert curve.apply_control_delta(xp, h, [0.1, 0.1, 0.1]) == xp


def test_curve_shape_is_preserved_under_uniform_delta():
    xp, h = make_label(25, 47, 0.5)
    for i in range(25, 48):
        xp[i] = 0.3 + 0.01 * (i - 25)       # 平坦でない曲線
    out = curve.apply_control_delta(xp, h, [0.1, 0.1, 0.1])
    for i in range(25, 48):
        assert out[i] == pytest.approx(xp[i] + 0.1, abs=1e-12)


def test_tilt_keeps_bottom_row_fixed():
    xp, h = make_label(25, 47, 0.5)
    out = curve.apply_tilt(xp, h, 0.2)
    assert out[47] == pytest.approx(0.5, abs=1e-12)


def test_tilt_moves_top_row_by_delta():
    xp, h = make_label(25, 47, 0.5)
    out = curve.apply_tilt(xp, h, 0.2)
    assert out[25] == pytest.approx(0.7, abs=1e-12)


def test_tilt_is_linear_in_between():
    xp, h = make_label(25, 47, 0.5)
    out = curve.apply_tilt(xp, h, 0.22)
    assert out[36] == pytest.approx(0.5 + 0.22 * (36 - 47) / (25 - 47), abs=1e-12)


def test_tilt_clips_to_unit_range():
    xp, h = make_label(25, 47, 0.95)
    out = curve.apply_tilt(xp, h, 0.5)
    assert max(out) <= 1.0


def test_tilt_leaves_invalid_rows_untouched():
    xp, h = make_label(25, 47, 0.5)
    xp[3] = 0.42
    out = curve.apply_tilt(xp, h, 0.2)
    assert out[3] == 0.42


def test_regenerate_passes_through_control_points():
    _, h = make_label(25, 47)
    out = curve.regenerate(h, [0.2, 0.6, 0.4])
    rows = curve.control_rows(h, 3)
    assert out[rows[0]] == pytest.approx(0.2, abs=1e-9)
    assert out[rows[1]] == pytest.approx(0.6, abs=1e-9)
    assert out[rows[2]] == pytest.approx(0.4, abs=1e-9)


def test_regenerate_zeroes_invalid_rows():
    _, h = make_label(25, 47)
    out = curve.regenerate(h, [0.2, 0.6, 0.4])
    assert out[0] == 0.0
    assert out[24] == 0.0


def test_regenerate_is_smooth_between_control_points():
    _, h = make_label(25, 47)
    out = curve.regenerate(h, [0.2, 0.6, 0.4])
    valid = [i for i, v in enumerate(h) if v]
    second = [out[i + 1] - 2 * out[i] + out[i - 1] for i in valid[1:-1]]
    # Catmull-Rom の 2 階差分は理論上 0.013 程度。ギザギザなら 0.8 前後になるので
    # 0.05 で「滑らか」と「壊れている」を十分に切り分けられる。
    assert max(abs(x) for x in second) < 0.05


def test_regenerate_with_two_control_points_is_linear():
    _, h = make_label(25, 47)
    out = curve.regenerate(h, [0.2, 0.6])
    assert out[36] == pytest.approx(0.2 + 0.4 * (36 - 25) / (47 - 25), abs=1e-9)


def test_regenerate_clips_to_unit_range():
    _, h = make_label(25, 47)
    out = curve.regenerate(h, [-0.5, 1.5, 0.5])
    assert min(out) >= 0.0 and max(out) <= 1.0


def test_interpolate_fills_rows_between_endpoints():
    xp, h = make_label(25, 47, 0.5)
    for i in range(30, 36):
        xp[i] = 0.0
        h[i] = 0
    out_xp, out_h = curve.interpolate_rows(xp, h, 29, 0.2, 36, 0.9)
    assert all(out_h[i] == 1 for i in range(29, 37))
    for i in range(29, 37):
        expected = 0.2 + (0.9 - 0.2) * (i - 29) / (36 - 29)
        assert out_xp[i] == pytest.approx(expected, abs=1e-12)


def test_interpolate_leaves_outside_rows_untouched():
    xp, h = make_label(25, 47, 0.5)
    xp[20] = 0.77
    out_xp, out_h = curve.interpolate_rows(xp, h, 30, 0.2, 35, 0.9)
    assert out_xp[20] == 0.77
    assert out_h[20] == 0
    assert out_xp[25] == 0.5 and out_xp[47] == 0.5


def test_interpolate_accepts_reversed_endpoints():
    xp, h = make_label(25, 47, 0.5)
    a = curve.interpolate_rows(xp, h, 30, 0.2, 40, 0.8)
    b = curve.interpolate_rows(xp, h, 40, 0.8, 30, 0.2)
    assert a[0] == b[0] and a[1] == b[1]


def test_interpolate_single_row():
    xp, h = make_label(25, 47, 0.5)
    h[33] = 0
    out_xp, out_h = curve.interpolate_rows(xp, h, 33, 0.42, 33, 0.42)
    assert out_h[33] == 1
    assert out_xp[33] == pytest.approx(0.42)


def test_interpolate_clips_to_unit_range():
    xp, h = make_label(25, 47, 0.5)
    out_xp, _ = curve.interpolate_rows(xp, h, 30, -0.5, 40, 1.5)
    assert min(out_xp) >= 0.0 and max(out_xp) <= 1.0


def test_interpolate_does_not_mutate_inputs():
    xp, h = make_label(25, 47, 0.5)
    before_xp, before_h = list(xp), list(h)
    curve.interpolate_rows(xp, h, 30, 0.2, 40, 0.8)
    assert xp == before_xp and h == before_h


def test_clear_rows_removes_points_in_range():
    xp, h = make_label(25, 47, 0.5)
    out_xp, out_h = curve.clear_rows(xp, h, 30, 35)
    for i in range(30, 36):
        assert out_h[i] == 0 and out_xp[i] == 0.0
    assert out_h[29] == 1 and out_xp[29] == 0.5
    assert out_h[36] == 1 and out_xp[36] == 0.5


def test_clear_rows_accepts_reversed_endpoints():
    xp, h = make_label(25, 47, 0.5)
    assert curve.clear_rows(xp, h, 35, 30) == curve.clear_rows(xp, h, 30, 35)


def test_clear_rows_does_not_mutate_inputs():
    xp, h = make_label(25, 47, 0.5)
    before_xp, before_h = list(xp), list(h)
    curve.clear_rows(xp, h, 30, 35)
    assert xp == before_xp and h == before_h
