"""
Derived module from pidmd.py for Physics-informed DMD with control.

Combines the manifold-constrained operator from PiDMD with the control
input handling from DMDc.

References:
- Peter J. Baddoo, Benjamin Herrmann, Beverley J. McKeon, J. Nathan Kutz, and
Steven L. Brunton. Physics-informed dynamic mode decomposition (pidmd). 2021.
arXiv:2112.04307.
- Proctor, J.L., Brunton, S.L. and Kutz, J.N., 2016. Dynamic mode decomposition
with control. SIAM Journal on Applied Dynamical Systems, 15(1), pp.142-161.
"""

import numpy as np

from .pidmd import PiDMD, PiDMDOperator
from .snapshots import Snapshots
from .utils import compute_svd, compute_tlsq


class PiDMDcOperator(PiDMDOperator):
    """
    DMD operator for Physics-informed DMD with control.

    Extends PiDMDOperator to handle control inputs. When B is known,
    the control contribution is subtracted before applying the manifold-
    constrained Procrustes solver. When B is unknown, an augmented SVD
    is used to separate state and control dynamics.

    :param manifold: the matrix manifold for the full DMD operator A.
    :type manifold: str
    :param manifold_opt: option used to specify certain manifolds.
    :type manifold_opt: int, tuple(int,int), or numpy.ndarray
    :param compute_A: Flag that determines whether or not to compute the
        full Koopman operator A.
    :type compute_A: bool
    :param svd_rank: the rank for the truncation.
    :type svd_rank: int or float
    :param svd_rank_omega: the rank for the truncation of the augmented
        matrix omega. Used only when B is unknown.
    :type svd_rank_omega: int or float
    """

    def __init__(
        self,
        manifold,
        manifold_opt,
        compute_A,
        svd_rank,
        svd_rank_omega=-1,
    ):
        super().__init__(
            manifold=manifold,
            manifold_opt=manifold_opt,
            compute_A=compute_A,
            svd_rank=svd_rank,
        )
        self._svd_rank_omega = svd_rank_omega
        self._B = None

    @property
    def B(self):
        """
        Get the control operator B.

        :return: the control operator B.
        :rtype: numpy.ndarray
        """
        if self._B is None:
            raise ValueError("You need to call fit before")
        return self._B

    def compute_operator_b_known(self, X, Y, B, controlin):
        """
        Compute the manifold-constrained operator when B is known.

        Subtracts the control contribution B*u from Y, then applies the
        manifold-constrained Procrustes solver from PiDMD.

        :param numpy.ndarray X: snapshots x0,..x{n-1} by column.
        :param numpy.ndarray Y: snapshots x1,..x{n} by column.
        :param numpy.ndarray B: the known control input matrix.
        :param numpy.ndarray controlin: the control input signals.
        :return: the (truncated) left-singular vectors matrix, the
            (truncated) singular values array, and the (truncated)
            right-singular vectors matrix of X.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """
        X, Y = compute_tlsq(X, Y, 0)
        Y_corrected = Y - B.dot(controlin)
        self._B = B
        return self.compute_operator(X, Y_corrected)

    def compute_operator_b_unknown(self, X, Y, controlin):
        """
        Compute the manifold-constrained operator when B is unknown.

        Uses an augmented SVD on [X; I] to separate state and control
        dynamics. B is recovered from the control subspace, then the
        manifold-constrained Procrustes solver is applied to recover A.

        :param numpy.ndarray X: snapshots x0,..x{n-1} by column.
        :param numpy.ndarray Y: snapshots x1,..x{n} by column.
        :param numpy.ndarray controlin: the control input signals.
        :return: the basis vectors used for dimension reduction.
        :rtype: numpy.ndarray
        """
        snapshots_rows = X.shape[0]

        # Augmented matrix: stack state snapshots and control inputs
        omega = np.vstack([X, controlin])

        # SVD of the augmented matrix
        Up, sp, Vp = compute_svd(omega, self._svd_rank_omega)

        # Split the left singular vectors
        Up1 = Up[:snapshots_rows, :]  # state subspace
        Up2 = Up[snapshots_rows:, :]  # control subspace

        # SVD of output for basis
        Ur, _, _ = compute_svd(Y, self._svd_rank)

        # Compute B-tilde in reduced space, then expand
        Btilde = np.linalg.multi_dot(
            [Ur.T.conj(), Y, Vp, np.diag(np.reciprocal(sp)), Up2.T.conj()]
        )
        self._B = Ur.dot(Btilde)

        # Correct Y by removing the control contribution
        Y_corrected = Y - self._B.dot(controlin)

        # Now apply the manifold-constrained Procrustes solver
        U, s, V = compute_svd(X, self._svd_rank)
        result_dict = self._compute_procrustes(X, Y_corrected)

        # Process the result (same logic as PiDMDOperator.compute_operator)
        if "atilde" in result_dict:
            self._Atilde = result_dict["atilde"]
            self._eigenvalues, self._eigenvectors = np.linalg.eig(
                self._Atilde
            )
            self._modes = U.dot(self._eigenvectors)
            if self._compute_A:
                self._A = np.linalg.multi_dot(
                    [
                        self._modes,
                        np.diag(self._eigenvalues),
                        np.linalg.pinv(self._modes),
                    ]
                )
        else:
            if "A" in result_dict:
                self._A = result_dict["A"]
                self._eigenvalues, self._modes = np.linalg.eig(self._A)
            else:
                self._eigenvalues = result_dict["eigenvalues"]
                self._modes = result_dict["modes"]
            self._eigenvectors = U.conj().T.dot(self._modes)
            self._Atilde = np.linalg.multi_dot(
                [
                    self._eigenvectors,
                    np.diag(self._eigenvalues),
                    np.linalg.pinv(self._eigenvectors),
                ]
            )

        return Ur


class PiDMDc(PiDMD):
    """
    Physics-informed Dynamic Mode Decomposition with control.

    Combines the manifold-constrained operator from PiDMD with the
    control input handling from DMDc. The system model is:

        x_{k+1} = A x_k + B u_k

    where A is constrained to lie on a specified matrix manifold and
    B is unconstrained.

    :param manifold: the matrix manifold to restrict the full operator
        A to. See :class:`PiDMD` for the list of supported manifolds.
    :type manifold: str
    :param manifold_opt: option used to specify certain manifolds.
        See :class:`PiDMD` for details.
    :type manifold_opt: int, tuple(int,int), or numpy.ndarray
    :param compute_A: Flag that determines whether or not to compute
        the full Koopman operator A.
    :type compute_A: bool
    :param svd_rank: the rank for the truncation; If 0, the method
        computes the optimal rank and uses it for truncation; if positive
        integer, the method uses the argument for the truncation; if
        float between 0 and 1, the rank is the number of the biggest
        singular values that are needed to reach the 'energy' specified
        by `svd_rank`; if -1, the method does not compute truncation.
        Default is -1.
    :type svd_rank: int or float
    :param svd_rank_omega: the rank for the truncation of the augmented
        matrix omega composed by the left snapshots matrix and the
        control input. Used only when B is unknown. Default is -1.
    :type svd_rank_omega: int or float
    :param tlsq_rank: rank truncation computing Total Least Square.
        Default is 0, meaning no truncation.
    :type tlsq_rank: int
    :param opt: If True, amplitudes are computed like in optimized DMD.
        Default is False.
    :type opt: bool or int
    :param lag: the time lag between snapshots. Default is 1.
    :type lag: int
    """

    def __init__(
        self,
        manifold,
        manifold_opt=None,
        compute_A=False,
        svd_rank=-1,
        svd_rank_omega=-1,
        tlsq_rank=0,
        opt=False,
        lag=1,
    ):
        # Call PiDMD.__init__ which sets up the operator
        super().__init__(
            manifold=manifold,
            manifold_opt=manifold_opt,
            compute_A=compute_A,
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
            opt=opt,
        )
        # Replace the operator with the control-aware version
        self._Atilde = PiDMDcOperator(
            manifold=manifold,
            manifold_opt=manifold_opt,
            compute_A=compute_A,
            svd_rank=svd_rank,
            svd_rank_omega=svd_rank_omega,
        )

        self._B = None
        self._controlin = None
        self._basis = None
        self._lag = lag

    @property
    def B(self):
        """
        Get the control operator B.

        :return: the control operator B.
        :rtype: numpy.ndarray
        """
        return self._B

    @property
    def basis(self):
        """
        Get the basis used to reduce the linear operator to the low
        dimensional space.

        :return: the matrix which columns are the basis vectors.
        :rtype: numpy.ndarray
        """
        return self._basis

    def fit(self, X, I, B=None):
        """
        Compute the Physics-informed Dynamic Mode Decomposition with
        control given the original snapshots and control input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        :param I: the control input.
        :type I: numpy.ndarray or iterable
        :param B: optional known control matrix. If None, B is computed
            from the data.
        :type B: numpy.ndarray or None
        """
        self._reset()
        self._controlin = np.atleast_2d(np.asarray(I))

        self._snapshots_holder = Snapshots(X)
        n_samples = self.snapshots.shape[-1]

        if self._lag < 1:
            raise ValueError("Time lag must be positive.")

        X_left = self.snapshots[:, : -self._lag]
        Y = self.snapshots[:, self._lag :]

        self._set_initial_time_dictionary(
            {"t0": 0, "tend": n_samples - 1, "dt": 1}
        )

        if B is not None:
            U, s, V = self.operator.compute_operator_b_known(
                X_left, Y, B, self._controlin
            )
            self._basis = U
            self._B = B
        else:
            self._basis = self.operator.compute_operator_b_unknown(
                X_left, Y, self._controlin
            )
            self._B = self.operator._B

        self._b = self._compute_amplitudes()

        return self

    def reconstructed_data(self, control_input=None):
        """
        Return the reconstructed data, computed using the
        `control_input` argument. If the `control_input` is not passed,
        the original input (in the `fit` method) is used.

        :param numpy.ndarray control_input: the input control matrix.
        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """
        controlin = (
            np.asarray(control_input)
            if control_input is not None
            else self._controlin
        )

        if controlin.shape[-1] != self.dynamics.shape[-1] - self._lag:
            raise RuntimeError(
                "The number of control inputs and the number of snapshots "
                "to reconstruct has to be the same"
            )

        eigs = np.power(
            self.eigs, self.dmd_time["dt"] // self.original_time["dt"]
        )

        pinv_modes = np.linalg.pinv(self.modes)
        data = [self.snapshots[:, i] for i in range(self._lag)]

        for i, u in enumerate(controlin.T):
            arr = np.linalg.multi_dot(
                [self.modes, np.diag(eigs), pinv_modes, data[i]]
            ) + self._B.dot(u)
            data.append(arr)

        return np.array(data).T
