from roboticstoolbox.models import Panda
import numpy as np
from spatialmath.base.argcheck import (
    isvector,
    getvector,
    getmatrix,
    getunit,
    verifymatrix,
)

class Panda_new(Panda):
    def __init__(self): 
        super().__init__()
    

    def jacobm(self, q=None, J=None, H=None, end=None, start=None, axes="all"):
        r"""
        Calculates the manipulability Jacobian. This measure relates the rate
        of change of the manipulability to the joint velocities of the robot.
        One of J or q is required. Supply J and H if already calculated to
        save computation time

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param J: The manipulator Jacobian in any frame
        :type J: float ndarray(6,n)
        :param H: The manipulator Hessian in any frame
        :type H: float ndarray(6,n,n)
        :param end: the final link or Gripper which the Hessian represents
        :type end: str or ELink or Gripper
        :param start: the first link which the Hessian represents
        :type start: str or ELink

        :return: The manipulability Jacobian
        :rtype: float ndarray(n)

        Yoshikawa's manipulability measure

        .. math::

            m(\vec{q}) = \sqrt{\mat{J}(\vec{q}) \mat{J}(\vec{q})^T}

        This method returns its Jacobian with respect to configuration

        .. math::

            \frac{\partial m(\vec{q})}{\partial \vec{q}}

        :references:
            - Kinematic Derivatives using the Elementary Transform
                Sequence, J. Haviland and P. Corke
        """

        end, start, _ = self._get_limit_links(end, start)
        # path, n, _ = self.get_path(end, start)

        if axes == "all":
            axes = [True, True, True, True, True, True]
        elif axes.startswith("trans"):
            axes = [True, True, True, False, False, False]
        elif axes.startswith("rot"):
            axes = [False, False, False, True, True, True]
        elif axes.startswith("MM"):
            axes = [False, False, True, True, True, False]
        else:
            raise ValueError("axes must be all, trans or rot")

        if J is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, self.n)

            J = self.jacob0(q, start=start, end=end)
        else:
            verifymatrix(J, (6, self.n))

        n = J.shape[1]

        if H is None:
            H = self.hessian0(J0=J, start=start, end=end)
        else:
            verifymatrix(H, (6, self.n, self.n))

        manipulability = self.manipulability(q, J=J, start=start, end=end, axes=axes)

        J = J[axes, :]
        H = H[:, axes, :]

        b = np.linalg.inv(J @ np.transpose(J))
        Jm = np.zeros((n, 1))

        for i in range(n):
            c = J @ np.transpose(H[i, :, :])
            Jm[i, 0] = manipulability * np.transpose(c.flatten("F")) @ b.flatten("F")

        return Jm
    

        cm = np.r_[-self.panda.jacobm_rot(axes=axes, Rot=Rot).reshape((na,)), np.zeros(nb+6)]

    def jacobm_rot(self, Rot, q=None, J=None, H=None, end=None, start=None, axes="all"):
        r"""
        Calculates the manipulability Jacobian. This measure relates the rate
        of change of the manipulability to the joint velocities of the robot.
        One of J or q is required. Supply J and H if already calculated to
        save computation time

        :param q: The joint angles/configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param J: The manipulator Jacobian in any frame
        :type J: float ndarray(6,n)
        :param H: The manipulator Hessian in any frame
        :type H: float ndarray(6,n,n)
        :param end: the final link or Gripper which the Hessian represents
        :type end: str or ELink or Gripper
        :param start: the first link which the Hessian represents
        :type start: str or ELink

        :return: The manipulability Jacobian
        :rtype: float ndarray(n)

        Yoshikawa's manipulability measure

        .. math::

            m(\vec{q}) = \sqrt{\mat{J}(\vec{q}) \mat{J}(\vec{q})^T}

        This method returns its Jacobian with respect to configuration

        .. math::

            \frac{\partial m(\vec{q})}{\partial \vec{q}}

        :references:
            - Kinematic Derivatives using the Elementary Transform
                Sequence, J. Haviland and P. Corke
        """

        end, start, _ = self._get_limit_links(end, start)
        # path, n, _ = self.get_path(end, start)

        if axes == "all":
            axes = [True, True, True, True, True, True]
        elif axes =="trans":
            axes = [True, True, True, False, False, False]
        elif axes =="rot":
            axes = [False, False, False, True, True, True]
        elif axes == "MM":
            axes = [False, False, True, True, True, False]
        elif type(axes) == type([False, False, True, True, True, False]):
            axes = axes
        else:
            raise ValueError("axes must be all, trans or rot")

        if J is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, self.n)

            J = self.jacob0(q, start=start, end=end)
        else:
            verifymatrix(J, (6, self.n))

        n = J.shape[1]

        if H is None:
            H = self.hessian0(J0=J, start=start, end=end)
        else:
            verifymatrix(H, (6, self.n, self.n))

        R_trans = np.r_[np.c_[Rot, np.zeros((3, 3))],
                np.c_[np.zeros((3, 3)), Rot]]
        
        J_prime = R_trans @ J
        H_prime = R_trans @ H

        manipulability = self.manipulability(q, J=J_prime, start=start, end=end, axes=axes)

        J_prime = J_prime[axes, :]
        H_prime = H_prime[:, axes, :]

        b = np.linalg.inv(J_prime @ np.transpose(J_prime))
        Jm = np.zeros((n, 1))

        for i in range(n):
            c = J_prime @ np.transpose(H_prime[i, :, :])
            Jm[i, 0] = manipulability * np.transpose(c.flatten("F")) @ b.flatten("F")

        return Jm
    

    def jacobf(self, w, u, q=None, J=None, H=None, end=None, start=None):

        end, start, _ = self._get_limit_links(end, start)
        # path, n, _ = self.get_path(end, start)

        # if axes == "all":
        #     axes = [True, True, True, True, True, True]
        # elif axes.startswith("trans"):
        #     axes = [True, True, True, False, False, False]
        # elif axes.startswith("rot"):
        #     axes = [False, False, False, True, True, True]
        # elif axes.startswith("MM"):
        #     axes = [False, False, True, True, True, False]
        # else:
        #     raise ValueError("axes must be all, trans or rot")

        if J is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, self.n)

            J = self.jacob0(q, start=start, end=end)
        else:
            verifymatrix(J, (6, self.n))

        n = J.shape[1]

        if H is None:
            H = self.hessian0(J0=J, start=start, end=end)
        else:
            verifymatrix(H, (self.n, 6, self.n))
        # H [n, 6, n]
        Jf = np.zeros((1, n))
        for i in range(n):
            c = 2 * w.T @ w @ J.T @ u @ u.T @ H[i, :, :]
            Jf[0, i] = np.trace(c)

        return Jf

    def jacobf_MM(self, w, u, R_w2a, nb, q=None, J=None, H=None, end=None, start=None):
        # u is in world frame
        end, start, _ = self._get_limit_links(end, start)
        R6_w2a = np.r_[np.c_[R_w2a, np.zeros((3, 3))],
                  np.c_[np.zeros((3, 3)), R_w2a]]
        # path, n, _ = self.get_path(end, start)

        if J is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, self.n)

            J = self.jacob0(q, start=start, end=end)
        else:
            verifymatrix(J, (6, self.n))

        if H is None:
            H = self.hessian0(J0=J, start=start, end=end)
        else:
            verifymatrix(H, (self.n, 6, self.n))
        # H [n, 6, n]
        Jf = np.zeros((1, self.n+nb))
        for i in range(self.n):
            c = 2 * w.T @ w @ J.T @ R6_w2a @ u @ u.T @ R6_w2a.T @ H[i, :, :]
            Jf[0, i] = np.trace(c)
        
        Jf[0, self.n] = 0
        Jf[0, self.n+1] = 0
        d_R_w2a = np.eye(3)
        d_R_w2a[0, 0] = R_w2a[1, 0]
        d_R_w2a[0, 1] = R_w2a[0, 0]
        d_R_w2a[1, 1] = R_w2a[1, 0]
        d_R_w2a[1, 0] = -R_w2a[0, 0]
        d_R6_w2a = np.r_[np.c_[d_R_w2a, np.zeros((3, 3))],
                  np.c_[np.zeros((3, 3)), d_R_w2a]]

        c = 2 * u @ u.T @ R6_w2a @ J @ w.T @ w @ J.T @ d_R6_w2a
        Jf[0, self.n+2] = np.trace(c)
        return Jf
    
    def forcebility(self, W, u, q=None, J=None, H=None, end=None, start=None):
        if J is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, self.n)

            J = self.jacob0(q, start=start, end=end)
        else:
            verifymatrix(J, (6, self.n))

        H_force = (u.T @ J @ W.T @ W @ J.T @ u) **(0.5)
        return H_force

    def forcebility_MM(self, W, u, R_w2a, q=None, J=None, H=None, end=None, start=None):
        R6_w2a = np.r_[np.c_[R_w2a, np.zeros((3, 3))],
            np.c_[np.zeros((3, 3)), R_w2a]]
        if J is None:
            if q is None:
                q = np.copy(self.q)
            else:
                q = getvector(q, self.n)

            J = self.jacob0(q, start=start, end=end)
        else:
            verifymatrix(J, (6, self.n))

        H_force = (u.T @ R6_w2a.T @ J @ W.T @ W @ J.T @ R6_w2a @ u) **(0.5)
        return H_force


    def manipulability(self, q=None, J=None, method="yoshikawa", axes="all", **kwargs):

        """
        Manipulability measure

        :param q: Joint coordinates, one of J or q required
        :type q: ndarray(n), or ndarray(m,n)
        :param J: Jacobian in world frame if already computed, one of J or
            q required
        :type J: ndarray(6,n)
        :param method: method to use, "yoshikawa" (default), "condition",
            "minsingular"  or "asada"
        :type method: str
        :param axes: Task space axes to consider: "all" [default],
            "trans", "rot" or "both"
        :type axes: str
        :param kwargs: extra arguments to pass to ``jacob0``
        :return: manipulability
        :rtype: float or ndarray(m)

        - ``manipulability(q)`` is the scalar manipulability index
          for the robot at the joint configuration ``q``.  It indicates
          dexterity, that is, how well conditioned the robot is for motion
          with respect to the 6 degrees of Cartesian motion.  The values is
          zero if the robot is at a singularity.

        Various measures are supported:

        +-------------------+-------------------------------------------------+
        | Measure           |       Description                               |
        +-------------------+-------------------------------------------------+
        | ``"yoshikawa"``   | Volume of the velocity ellipsoid, *distance*    |
        |                   | from singularity [Yoshikawa85]_                 |
        +-------------------+-------------------------------------------------+
        | ``"invcondition"``| Inverse condition number of Jacobian, isotropy  |
        |                   | of the velocity ellipsoid [Klein87]_            |
        +-------------------+-------------------------------------------------+
        | ``"minsingular"`` | Minimum singular value of the Jacobian,         |
        |                   | *distance*  from singularity [Klein87]_         |
        +-------------------+-------------------------------------------------+
        | ``"asada"``       | Isotropy of the task-space acceleration         |
        |                   | ellipsoid which is a function of the Cartesian  |
        |                   | inertia matrix which depends on the inertial    |
        |                   | parameters [Asada83]_                           |
        +-------------------+-------------------------------------------------+

        **Trajectory operation**:

        If ``q`` is a matrix (m,n) then the result (m,) is a vector of
        manipulability indices for each joint configuration specified by a row
        of ``q``.

        .. note::

            - Invokes the ``jacob0`` method of the robot if ``J`` is not passed
            - The "all" option includes rotational and translational
              dexterity, but this involves adding different units. It can be
              more useful to look at the translational and rotational
              manipulability separately.
            - Examples in the RVC book (1st edition) can be replicated by
              using the "all" option
            - Asada's measure requires inertial a robot model with inertial
              parameters.

        :references:

        .. [Yoshikawa85] Manipulability of Robotic Mechanisms. Yoshikawa T.,
                The International Journal of Robotics Research.
                1985;4(2):3-9. doi:10.1177/027836498500400201
        .. [Asada83] A geometrical representation of manipulator dynamics and
                its application to arm design, H. Asada,
                Journal of Dynamic Systems, Measurement, and Control,
                vol. 105, p. 131, 1983.
        .. [Klein87] Dexterity Measures for the Design and Control of
                Kinematically Redundant Manipulators. Klein CA, Blaho BE.
                The International Journal of Robotics Research.
                1987;6(2):72-83. doi:10.1177/027836498700600206

        - Robotics, Vision & Control, Chap 8, P. Corke, Springer 2011.

        """
        if isinstance(axes, list) and len(axes) == 6:
            pass
        elif axes == "all":
            axes = [True, True, True, True, True, True]
        elif axes.startswith("trans"):
            axes = [True, True, True, False, False, False]
        elif axes.startswith("rot"):
            axes = [False, False, False, True, True, True]
        elif axes.startswith("MM"):
            axes = [False, False, True, True, True, False]
        elif axes == "both":
            return (
                self.manipulability(q, J, method, axes="trans", **kwargs),
                self.manipulability(q, J, method, axes="rot", **kwargs),
            )
        elif type(axes) != type([True]):
            axes = axes
        else:
            raise ValueError("axes must be all, trans, rot or both")

        def yoshikawa(robot, J, q, axes, **kwargs):
            J = J[axes, :]
            if J.shape[0] == J.shape[1]:
                # simplified case for square matrix
                return abs(np.linalg.det(J))
            else:
                m2 = np.linalg.det(J @ J.T)
                return np.sqrt(abs(m2))

        def condition(robot, J, q, axes, **kwargs):
            J = J[axes, :]
            return 1 / np.linalg.cond(J)  # return 1/cond(J)

        def minsingular(robot, J, q, axes, **kwargs):
            J = J[axes, :]
            s = np.linalg.svd(J, compute_uv=False)
            return s[-1]  # return last/smallest singular value of J

        def asada(robot, J, q, axes, **kwargs):
            # dof = np.sum(axes)
            if np.linalg.matrix_rank(J) < 6:
                return 0
            Ji = np.linalg.pinv(J)
            Mx = Ji.T @ robot.inertia(q) @ Ji
            d = np.where(axes)[0]
            Mx = Mx[d]
            Mx = Mx[:, d.tolist()]
            e, _ = np.linalg.eig(Mx)
            return np.min(e) / np.max(e)

        # choose the handler function
        if method == "yoshikawa":
            mfunc = yoshikawa
        elif method == "invcondition":
            mfunc = condition
        elif method == "minsingular":
            mfunc = minsingular
        elif method == "asada":
            mfunc = asada
        else:
            raise ValueError("Invalid method chosen")

        # Calculate manipulability based on supplied Jacobian
        if J is not None:
            w = [mfunc(self, J, q, axes)]

        # Otherwise use the q vector/matrix
        else:
            q = getmatrix(q, (None, self.n))
            w = np.zeros(q.shape[0])

            for k, qk in enumerate(q):
                Jk = self.jacob0(qk, **kwargs)
                w[k] = mfunc(self, Jk, qk, axes)

        if len(w) == 1:
            return w[0]
        else:
            return w

