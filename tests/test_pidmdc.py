import numpy as np
import scipy
from numpy.testing import assert_allclose
from pytest import raises

from pydmd.pidmdc import PiDMDc


def error(true, comp):
    """Helper function that computes and returns relative error."""
    return np.linalg.norm(comp - true) / np.linalg.norm(true)


# ---------------------------------------------------------------------------
# Test system generators
# ---------------------------------------------------------------------------


def create_system_with_B():
    """Create a simple controlled system with known B."""
    rng = np.random.default_rng(seed=42)
    n = 10  # state dimension
    m = 3  # control dimension
    nt = 50  # number of time steps

    # Unitary A (conservative dynamics)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Q
    B = rng.standard_normal((n, m))

    x0 = rng.standard_normal(n)
    u = 0.1 * rng.standard_normal((m, nt - 1))
    snapshots = [x0]
    for i in range(nt - 1):
        snapshots.append(A.dot(snapshots[i]) + B.dot(u[:, i]))
    snapshots = np.array(snapshots).T

    return {"snapshots": snapshots, "u": u, "B": B, "A": A}


def create_system_without_B():
    """Create a controlled system where B must be learned."""
    rng = np.random.default_rng(seed=123)
    n = 5  # state dimension
    m = 15  # number of time steps

    A = scipy.linalg.helmert(n, True)
    B = rng.standard_normal((n, n)) - 0.5
    x0 = np.array([0.25] * n)
    u = rng.standard_normal((n, m - 1)) - 0.5
    snapshots = [x0]
    for i in range(m - 1):
        snapshots.append(A.dot(snapshots[i]) + B.dot(u[:, i]))
    snapshots = np.array(snapshots).T

    return {"snapshots": snapshots, "u": u, "B": B, "A": A}


def create_unitary_system_with_noise(noise_mag=0.0):
    """Create a unitary system with optional noise for robustness tests."""
    rng = np.random.default_rng(seed=99)
    n = 8
    m = 2
    nt = 200

    # Unitary rotation matrix
    theta = np.pi / 6
    A_block = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    A = scipy.linalg.block_diag(*[A_block] * (n // 2))
    B = rng.standard_normal((n, m)) * 0.5

    x0 = rng.standard_normal(n)
    u = 0.05 * rng.standard_normal((m, nt - 1))
    snapshots = [x0]
    for i in range(nt - 1):
        snapshots.append(A.dot(snapshots[i]) + B.dot(u[:, i]))
    snapshots = np.array(snapshots).T

    if noise_mag > 0:
        snapshots += noise_mag * rng.standard_normal(snapshots.shape)

    return {"snapshots": snapshots, "u": u, "B": B, "A": A}


# ---------------------------------------------------------------------------
# Helper assertions
# ---------------------------------------------------------------------------


def assert_all_zero(A):
    """Assert that A is approximately zero."""
    assert_allclose(np.linalg.norm(A), 0, atol=1e-10)


def assert_circulant(A):
    """Assert that A is a circulant matrix."""
    for i in range(1, len(A)):
        assert_allclose(np.roll(A[i, :], -i), A[0, :], atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: invalid inputs
# ---------------------------------------------------------------------------


def test_invalid_manifold():
    system = create_system_with_B()
    with raises(ValueError):
        PiDMDc("invalid_manifold").fit(
            system["snapshots"], system["u"], system["B"]
        )


def test_invalid_lag():
    system = create_system_with_B()
    pidmdc = PiDMDc("unitary", lag=0)
    with raises(ValueError):
        pidmdc.fit(system["snapshots"], system["u"], system["B"])


# ---------------------------------------------------------------------------
# Tests: B known — manifold structure is preserved
# ---------------------------------------------------------------------------


def test_unitary_b_known():
    system = create_system_with_B()
    pidmdc = PiDMDc("unitary", compute_A=True).fit(
        system["snapshots"], system["u"], system["B"]
    )
    # A should be unitary
    n = system["A"].shape[0]
    assert_allclose(
        pidmdc.A.conj().T.dot(pidmdc.A), np.eye(n), atol=1e-10
    )


def test_symmetric_b_known():
    system = create_system_with_B()
    pidmdc = PiDMDc("symmetric", compute_A=True).fit(
        system["snapshots"], system["u"], system["B"]
    )
    assert_allclose(pidmdc.A, pidmdc.A.T, atol=1e-10)


def test_skewsymmetric_b_known():
    system = create_system_with_B()
    pidmdc = PiDMDc("skewsymmetric", compute_A=True).fit(
        system["snapshots"], system["u"], system["B"]
    )
    assert_allclose(pidmdc.A, -pidmdc.A.T, atol=1e-10)


def test_uppertriangular_b_known():
    system = create_system_with_B()
    pidmdc = PiDMDc("uppertriangular", compute_A=True).fit(
        system["snapshots"], system["u"], system["B"]
    )
    assert_all_zero(np.tril(pidmdc.A, k=-1))


def test_lowertriangular_b_known():
    system = create_system_with_B()
    pidmdc = PiDMDc("lowertriangular", compute_A=True).fit(
        system["snapshots"], system["u"], system["B"]
    )
    assert_all_zero(np.triu(pidmdc.A, k=1))


def test_diagonal_b_known():
    system = create_system_with_B()
    pidmdc = PiDMDc(
        "diagonal", manifold_opt=1, compute_A=True
    ).fit(system["snapshots"], system["u"], system["B"])
    assert_all_zero(pidmdc.A - np.diag(np.diag(pidmdc.A)))


# ---------------------------------------------------------------------------
# Tests: B known — reconstruction
# ---------------------------------------------------------------------------


def test_reconstruct_b_known():
    system = create_unitary_system_with_noise(noise_mag=0.0)
    pidmdc = PiDMDc("unitary", svd_rank=-1).fit(
        system["snapshots"], system["u"], system["B"]
    )
    reconstructed = pidmdc.reconstructed_data()
    err = error(system["snapshots"], reconstructed)
    assert err < 0.1, f"Reconstruction error too high: {err}"


def test_B_stored_b_known():
    system = create_system_with_B()
    pidmdc = PiDMDc("unitary").fit(
        system["snapshots"], system["u"], system["B"]
    )
    np.testing.assert_array_equal(pidmdc.B, system["B"])


# ---------------------------------------------------------------------------
# Tests: B unknown
# ---------------------------------------------------------------------------


def test_eigs_b_unknown():
    system = create_system_without_B()
    pidmdc = PiDMDc(
        "unitary", svd_rank=3, svd_rank_omega=4
    ).fit(system["snapshots"], system["u"])
    assert pidmdc.eigs.shape[0] == 3


def test_modes_b_unknown():
    system = create_system_without_B()
    pidmdc = PiDMDc(
        "unitary", svd_rank=3, svd_rank_omega=4
    ).fit(system["snapshots"], system["u"])
    assert pidmdc.modes.shape[1] == 3


def test_B_computed_b_unknown():
    system = create_system_without_B()
    pidmdc = PiDMDc(
        "unitary", svd_rank=-1, svd_rank_omega=-1
    ).fit(system["snapshots"], system["u"])
    assert pidmdc.B is not None
    assert pidmdc.B.shape == system["B"].shape


def test_reconstruct_b_unknown():
    system = create_system_without_B()
    pidmdc = PiDMDc(
        "unitary", svd_rank=-1, svd_rank_omega=-1, opt=True
    ).fit(system["snapshots"], system["u"])
    reconstructed = pidmdc.reconstructed_data()
    err = error(system["snapshots"], reconstructed)
    assert err < 0.5, f"Reconstruction error too high: {err}"


def test_symmetric_b_unknown():
    system = create_system_without_B()
    pidmdc = PiDMDc(
        "symmetric", svd_rank=-1, svd_rank_omega=-1, compute_A=True
    ).fit(system["snapshots"], system["u"])
    assert_allclose(pidmdc.A, pidmdc.A.T, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: reconstruction with new control inputs
# ---------------------------------------------------------------------------


def test_reconstruct_new_control():
    system = create_unitary_system_with_noise(noise_mag=0.0)
    pidmdc = PiDMDc("unitary", svd_rank=-1).fit(
        system["snapshots"], system["u"], system["B"]
    )
    # Use different control inputs for reconstruction
    rng = np.random.default_rng(seed=777)
    new_u = 0.05 * rng.standard_normal(system["u"].shape)
    reconstructed = pidmdc.reconstructed_data(new_u)
    assert reconstructed.shape == system["snapshots"].shape


def test_reconstruct_wrong_control_size():
    system = create_system_with_B()
    pidmdc = PiDMDc("unitary").fit(
        system["snapshots"], system["u"], system["B"]
    )
    wrong_u = np.zeros((system["u"].shape[0], 2))
    with raises(RuntimeError):
        pidmdc.reconstructed_data(wrong_u)


# ---------------------------------------------------------------------------
# Tests: properties and basic interface
# ---------------------------------------------------------------------------


def test_basis_exists():
    system = create_system_with_B()
    pidmdc = PiDMDc("unitary").fit(
        system["snapshots"], system["u"], system["B"]
    )
    assert pidmdc.basis is not None


def test_A_property():
    system = create_system_with_B()
    pidmdc = PiDMDc("unitary", compute_A=True).fit(
        system["snapshots"], system["u"], system["B"]
    )
    assert pidmdc.A.shape[0] == pidmdc.A.shape[1]


def test_A_not_computed_raises():
    system = create_system_with_B()
    pidmdc = PiDMDc("unitary", compute_A=False).fit(
        system["snapshots"], system["u"], system["B"]
    )
    with raises(ValueError):
        _ = pidmdc.A


def test_compute_A_required_for_triangular():
    system = create_system_with_B()
    with raises(ValueError):
        PiDMDc("uppertriangular", compute_A=False).fit(
            system["snapshots"], system["u"], system["B"]
        )


# ---------------------------------------------------------------------------
# Tests: noise robustness — PiDMDc vs DMDc
# ---------------------------------------------------------------------------


def test_noise_robustness():
    """PiDMDc with unitary manifold should be more robust to noise."""
    from pydmd import DMDc

    system = create_unitary_system_with_noise(noise_mag=1e-3)

    pidmdc = PiDMDc("unitary", svd_rank=-1).fit(
        system["snapshots"], system["u"], system["B"]
    )
    dmdc = DMDc(svd_rank=-1).fit(
        system["snapshots"], system["u"], system["B"]
    )

    # Both should produce eigenvalues; PiDMDc's should be on the unit circle
    pidmdc_eig_mags = np.abs(pidmdc.eigs)
    assert_allclose(pidmdc_eig_mags, np.ones_like(pidmdc_eig_mags), atol=1e-10)

    # DMDc eigenvalues may drift off the unit circle due to noise
    dmdc_eig_mags = np.abs(dmdc.eigs)
    # Not asserting DMDc fails, just that PiDMDc preserves the constraint
    assert np.max(np.abs(pidmdc_eig_mags - 1.0)) <= np.max(
        np.abs(dmdc_eig_mags - 1.0)
    )


# ---------------------------------------------------------------------------
# Tests: import from package
# ---------------------------------------------------------------------------


def test_import_from_pydmd():
    from pydmd import PiDMDc as PiDMDc_pkg

    assert PiDMDc_pkg is PiDMDc
