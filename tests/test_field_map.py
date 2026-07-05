import numpy as np
import pytest
from pydantic import ValidationError

from latent_dirac.fields.field_map import FieldMapField, load_comsol_grid_csv


def linear_b(points: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            2.0 * points[:, 0] + 0.1,
            3.0 * points[:, 1] - 0.2,
            -1.0 * points[:, 2],
        ]
    )


def make_linear_field_map() -> FieldMapField:
    x = np.linspace(-0.1, 0.1, 5)
    y = np.linspace(-0.05, 0.05, 4)
    z = np.linspace(0.0, 0.4, 6)
    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
    b_values = linear_b(grid.reshape(-1, 3)).reshape(5, 4, 6, 3)
    return FieldMapField(x_m=x, y_m=y, z_m=z, B_t=b_values)


def test_trilinear_interpolation_is_exact_for_linear_fields():
    field = make_linear_field_map()
    queries = np.array(
        [
            [0.013, -0.021, 0.137],
            [-0.08, 0.04, 0.31],
            [0.0, 0.0, 0.2],
        ]
    )

    np.testing.assert_allclose(field.B(queries, np.zeros(3)), linear_b(queries), rtol=1e-12)


def test_single_position_shape_matches_solver_contract():
    field = make_linear_field_map()
    point = np.array([0.013, -0.021, 0.137])

    result = field.B(point, 0.0)
    assert result.shape == (3,)
    np.testing.assert_allclose(result, linear_b(point[np.newaxis])[0], rtol=1e-12)


def test_queries_outside_grid_return_zero():
    field = make_linear_field_map()
    outside = np.array(
        [
            [0.2, 0.0, 0.2],  # x beyond max
            [0.0, -0.06, 0.2],  # y below min
            [0.0, 0.0, 0.5],  # z beyond max
        ]
    )

    np.testing.assert_allclose(field.B(outside, np.zeros(3)), np.zeros((3, 3)))


def test_electric_field_defaults_to_zero_and_interpolates_when_given():
    field = make_linear_field_map()
    np.testing.assert_allclose(field.E(np.array([0.0, 0.0, 0.2]), 0.0), np.zeros(3))

    with_e = FieldMapField(
        x_m=field.x_m,
        y_m=field.y_m,
        z_m=field.z_m,
        B_t=field.B_t,
        E_v_m=2.0 * field.B_t,
    )
    query = np.array([[0.013, -0.021, 0.137]])
    np.testing.assert_allclose(with_e.E(query, np.zeros(1)), 2.0 * linear_b(query), rtol=1e-12)


def test_non_increasing_axes_are_rejected():
    field = make_linear_field_map()
    bad_x = np.array(field.x_m)
    bad_x[1] = bad_x[0]

    with pytest.raises(ValidationError):
        FieldMapField(x_m=bad_x, y_m=field.y_m, z_m=field.z_m, B_t=field.B_t)


def test_shape_mismatch_is_rejected():
    field = make_linear_field_map()

    with pytest.raises(ValidationError):
        FieldMapField(x_m=field.x_m, y_m=field.y_m, z_m=field.z_m, B_t=field.B_t[:-1])


COMSOL_STYLE_DOCUMENT = """% Model: solenoid_fringe.mph
% Version: COMSOL 6.2
% Date: Jul 5 2026
% x, y, z, Bx, By, Bz
0.0,0.0,0.0,0.0,0.0,0.5
0.1,0.0,0.0,0.0,0.0,0.6
0.0,0.1,0.0,0.0,0.0,0.7
0.1,0.1,0.0,0.0,0.0,0.8
0.0,0.0,0.2,0.1,0.0,0.9
0.1,0.0,0.2,0.1,0.0,1.0
0.0,0.1,0.2,0.1,0.0,1.1
0.1,0.1,0.2,0.1,0.0,1.2
"""


def test_comsol_csv_import_builds_expected_grid(tmp_path):
    path = tmp_path / "map.csv"
    path.write_text(COMSOL_STYLE_DOCUMENT, encoding="utf-8")

    field = load_comsol_grid_csv(path)

    np.testing.assert_allclose(field.x_m, [0.0, 0.1])
    np.testing.assert_allclose(field.y_m, [0.0, 0.1])
    np.testing.assert_allclose(field.z_m, [0.0, 0.2])
    np.testing.assert_allclose(field.B(np.array([0.1, 0.1, 0.2]), 0.0), [0.1, 0.0, 1.2])
    np.testing.assert_allclose(field.B(np.array([0.0, 0.0, 0.0]), 0.0), [0.0, 0.0, 0.5])


def test_comsol_csv_import_accepts_shuffled_rows_and_whitespace(tmp_path):
    lines = [line for line in COMSOL_STYLE_DOCUMENT.strip().splitlines() if not line.startswith("%")]
    shuffled = [lines[i] for i in (5, 0, 7, 2, 4, 6, 1, 3)]
    document = "% header\n" + "\n".join(line.replace(",", " ") for line in shuffled) + "\n"
    path = tmp_path / "map.txt"
    path.write_text(document, encoding="utf-8")

    field = load_comsol_grid_csv(path)

    np.testing.assert_allclose(field.B(np.array([0.1, 0.1, 0.2]), 0.0), [0.1, 0.0, 1.2])


def test_comsol_csv_import_rejects_incomplete_grid(tmp_path):
    truncated = "\n".join(COMSOL_STYLE_DOCUMENT.strip().splitlines()[:-1]) + "\n"
    path = tmp_path / "map.csv"
    path.write_text(truncated, encoding="utf-8")

    with pytest.raises(ValueError):
        load_comsol_grid_csv(path)


def test_non_finite_field_values_are_rejected():
    field = make_linear_field_map()
    bad_values = np.array(field.B_t)
    bad_values[0, 0, 0, 2] = np.nan

    with pytest.raises(ValidationError):
        FieldMapField(x_m=field.x_m, y_m=field.y_m, z_m=field.z_m, B_t=bad_values)


def test_comsol_csv_import_rejects_nan_rows(tmp_path):
    document = COMSOL_STYLE_DOCUMENT.replace("0.1,0.1,0.2,0.1,0.0,1.2", "0.1,0.1,0.2,NaN,NaN,NaN")
    path = tmp_path / "map.csv"
    path.write_text(document, encoding="utf-8")

    with pytest.raises((ValueError, ValidationError)):
        load_comsol_grid_csv(path)


def test_comsol_csv_import_rejects_non_si_unit_headers(tmp_path):
    document = "% x (mm), y (mm), z (mm), Bx (T), By (T), Bz (T)\n" + "\n".join(
        line for line in COMSOL_STYLE_DOCUMENT.splitlines() if not line.startswith("%")
    )
    path = tmp_path / "map.csv"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="SI"):
        load_comsol_grid_csv(path)


def test_uniform_field_map_matches_uniform_field_transport():
    from examples.charge_sign_splitter_demo import make_initial_pair
    from latent_dirac.fields.uniform import UniformField
    from latent_dirac.solvers.relativistic_boris import RelativisticBorisSolver

    b_vector = np.array([0.0, 0.45, 0.0])
    x = np.linspace(-1.0, 1.0, 3)
    b_values = np.broadcast_to(b_vector, (3, 3, 3, 3)).copy()
    mapped = FieldMapField(x_m=x, y_m=x, z_m=x, B_t=b_values)
    analytic = UniformField(B_vector_t=b_vector)

    cloud, _ = make_initial_pair(particle_count=16, seed=7)
    solver = RelativisticBorisSolver(dt_s=2.0e-12, steps=40)

    np.testing.assert_allclose(
        solver.propagate(cloud, mapped).position_m,
        solver.propagate(cloud, analytic).position_m,
        rtol=1e-12,
    )


def test_render_field_magnitude_3d_smoke():
    pytest.importorskip("plotly")
    from latent_dirac.viz.field_3d import render_field_magnitude_3d

    figure = render_field_magnitude_3d(make_linear_field_map())

    assert len(figure.data) >= 1
    hover_or_name = " ".join(str(getattr(trace, "hovertext", "")) + str(trace.name) for trace in figure.data)
    assert "table-based" in hover_or_name
