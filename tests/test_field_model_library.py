import numpy as np
import pytest

from latent_dirac.fields.composite import CompositeField
from latent_dirac.fields.dipole import DipoleField
from latent_dirac.fields.quadrupole import QuadrupoleField
from latent_dirac.fields.uniform import UniformField


def test_composite_field_sums_electric_and_magnetic_components():
    field = CompositeField(
        fields=[
            UniformField(
                E_vector_v_m=np.array([1.0, 0.0, 2.0]),
                B_vector_t=np.array([0.0, 0.1, 0.0]),
            ),
            UniformField(
                E_vector_v_m=np.array([0.5, -1.0, 0.0]),
                B_vector_t=np.array([0.2, 0.1, -0.3]),
            ),
        ]
    )
    positions = np.array([[0.0, 0.0, 0.0], [0.01, -0.02, 0.5]])
    times = np.zeros(2)

    np.testing.assert_allclose(field.E(positions, times), [[1.5, -1.0, 2.0]] * 2)
    np.testing.assert_allclose(field.B(positions, times), [[0.2, 0.2, -0.3]] * 2)


def test_composite_field_supports_single_position():
    field = CompositeField(
        fields=[
            UniformField(B_vector_t=np.array([0.0, 0.0, 0.4])),
            UniformField(B_vector_t=np.array([0.0, 0.0, 0.6])),
        ]
    )

    np.testing.assert_allclose(field.B(np.zeros(3), 0.0), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(field.E(np.zeros(3), 0.0), [0.0, 0.0, 0.0])


def test_dipole_field_is_uniform_inside_hard_edge_and_zero_outside():
    field = DipoleField(B_vector_t=[0.0, 0.35, 0.0], length_m=0.2, center_z_m=0.1)
    positions = np.array(
        [
            [0.0, 0.0, 0.1],  # center: inside
            [0.02, -0.01, 0.0],  # z edge: inside (inclusive)
            [0.0, 0.0, 0.21],  # past far edge: outside
            [0.0, 0.0, -0.05],  # before near edge: outside
        ]
    )
    times = np.zeros(4)

    expected = np.array(
        [
            [0.0, 0.35, 0.0],
            [0.0, 0.35, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(field.B(positions, times), expected)
    np.testing.assert_allclose(field.E(positions, times), np.zeros((4, 3)))


def test_dipole_field_single_position_inside_and_outside():
    field = DipoleField(B_vector_t=[0.0, 0.35, 0.0], length_m=0.2)

    np.testing.assert_allclose(field.B(np.zeros(3), 0.0), [0.0, 0.35, 0.0])
    np.testing.assert_allclose(field.B(np.array([0.0, 0.0, 0.3]), 0.0), np.zeros(3))


def test_dipole_field_rejects_bad_inputs():
    with pytest.raises(ValueError):
        DipoleField(B_vector_t=[0.0, 0.35], length_m=0.2)
    with pytest.raises(ValueError):
        DipoleField(B_vector_t=[0.0, 0.35, 0.0], length_m=0.0)


def test_quadrupole_field_is_linear_at_known_positions():
    field = QuadrupoleField(gradient_t_m=8.0, length_m=0.1)
    positions = np.array(
        [
            [0.01, 0.02, 0.0],
            [-0.03, 0.0, 0.04],
            [0.0, 0.0, 0.0],
        ]
    )
    times = np.zeros(3)

    expected = np.array(
        [
            [8.0 * 0.02, 8.0 * 0.01, 0.0],
            [0.0, 8.0 * -0.03, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(field.B(positions, times), expected)
    np.testing.assert_allclose(field.E(positions, times), np.zeros((3, 3)))


def test_quadrupole_field_is_zero_outside_hard_edge():
    field = QuadrupoleField(gradient_t_m=8.0, length_m=0.1, center_z_m=0.2)

    np.testing.assert_allclose(field.B(np.array([0.01, 0.02, 0.0]), 0.0), np.zeros(3))
    np.testing.assert_allclose(field.B(np.array([0.01, 0.02, 0.2]), 0.0), [0.16, 0.08, 0.0])


def test_quadrupole_gradient_sign_swaps_focusing_planes():
    focusing = QuadrupoleField(gradient_t_m=8.0, length_m=0.1)
    defocusing = QuadrupoleField(gradient_t_m=-8.0, length_m=0.1)
    position = np.array([0.01, 0.02, 0.0])

    np.testing.assert_allclose(focusing.B(position, 0.0), -defocusing.B(position, 0.0))


def test_quadrupole_field_rejects_nonpositive_length():
    with pytest.raises(ValueError):
        QuadrupoleField(gradient_t_m=8.0, length_m=-0.1)


def test_composite_field_works_with_solver_contract_shapes():
    composite = CompositeField(
        fields=[
            DipoleField(B_vector_t=[0.0, 0.35, 0.0], length_m=0.2),
            QuadrupoleField(gradient_t_m=8.0, length_m=0.1, center_z_m=0.2),
        ]
    )
    positions = np.array([[0.01, 0.02, 0.0], [0.01, 0.02, 0.2]])
    times = np.zeros(2)

    expected = np.array(
        [
            [0.0, 0.35, 0.0],
            [0.16, 0.08, 0.0],
        ]
    )
    np.testing.assert_allclose(composite.B(positions, times), expected)
