import datetime
import itertools
from visupport import *
from vimod import *
from hdm import *


class DynamicExpectationMaximisation:
    """Model inversion with DEM."""

    def __init__(self, hdm):
        """Doc string."""
        self.Y = None
        self.tY = 0
        self.CEu = []  # conditional estimate <q(u)>
        self.CEP = []  # conditional estimate <q(P)>
        self.CEH = []  # conditional estimate <q(H)>

        self.hdm = hdm
        self.LT = self.hdm.L[0]
        self.LP = self.hdm.L[1]
        self.LH = self.hdm.L[2]
        self._dt_ = sym_shasf('dt', 1)
        self._msg_invert_ = ""  # keep message from last inversion

        EOX = np.cast['int32'](self.hdm.EMBED_ORDER_X)
        EOV = np.cast['int32'](self.hdm.EMBED_ORDER_V)

        self._num_ydim_ = np.sum(self.hdm.ydim).astype('int32')
        self._num_xdim_ = np.sum(self.hdm.xdim).astype('int32')
        self._num_vdim_ = np.sum(self.hdm.vdim).astype('int32')
        self._num_edim_ = np.sum(self.hdm.edim).astype('int32')
        self._num_Pdim_ = np.sum(self.hdm.Pdim).astype('int32')
        self._num_Hdim_ = np.sum(self.hdm.Hdim).astype('int32')

        self._num_ydim_embed_ = self._num_ydim_ * EOX
        self._num_edim_embed_ = self._num_edim_ * EOV
        self._num_xdim_embed_ = self._num_xdim_ * EOX
        self._num_vdim_embed_ = self._num_vdim_ * EOV
        self._num_udim_embed_ = self._num_xdim_embed_ + self._num_vdim_embed_

        # conditional covariances 𝚲 -> q(u, P, H)
        # start with initial guess [Cu/CP/CH],
        # subseqently updated with curvatures of internal energies:
        # (_Cu_/_CP_/_CH_)
        # inv(𝚲u) = Cu <- - inv(dLT/duu)
        # inv(𝚲P) = CP <- - inv(dLP/dPP) - inv(int[dLT/dPP]dt)
        # inv(𝚲H) = CH <- - inv(dLH/dHH) - inv(int[dLT/dHH]dt)
        self.Cu = sym_shamf("COVARIANCE M/u [m=1+; fmat]")
        self.CP = sym_shamf("COVARIANCE M/P [m=1+; fmat]")
        self.CH = sym_shamf("COVARIANCE M/H [m=1+; fmat]")
        self._Cu_ = None
        self._CP_ = None
        self._CH_ = None

        # gradients and curvatures of variational actions
        # of P and H. Accumulated at the end of u updates.
        # LAP(u) = LT + 0.5 * WP + 0.5 * WH
        # LAP(P) = LT + 0.5 * Wu + 0.5 * WH
        # LAP(H) = LT + 0.5 * Wu + 0.5 * WP
        # Wu     = tr[Cu * dLT/duu]
        # WP     = tr[CP * dLT/dPP]
        # WH     = tr[CH * dLT/dHH]
        #
        # ----LT------LP------LH------Wu------WP-------WH------
        #  .
        #  u  dLTdu   ~~~     ~~~     ~~~     dWPdu    dWHdu
        #  .
        # uu  dLTduu  ~~~     ~~~     ~~~     dWPduu   dWHduu
        #  .
        # uy  dLTduy  ~~~     ~~~     ~~~     (dWPduy) (dWHduy)
        #  .
        # ue  dLTdue  ~~~     ~~~     ~~~     (dWPdue) (dWPdue)
        #  .
        #  P  dLTdP   dLPdP   ~~~     dWudP   ~~~      (dWHdP )
        #  .
        # PP  dLTdPP  dLPdPP  ~~~     dWudPP  ~~~      (dWHdPP)
        #  .
        #  H  dLTdH   ~~~     dLHdH   dWudH   (dWPdH ) ~~~
        #  .
        # HH  dLTdHH  ~~~     dLHdHH  dWudHH  (dWPdHH) ~~~
        #  .

        #
        self._dEdP_   = None  # d(x - f)/dP
        self._dBdP_   = None  # d(v - g)/dP
        self._dEdu_   = None
        self._dBdu_   = None
        self._dEdy_   = None
        self._dBdy_   = None
        self._dEde_   = None
        self._dBde_   = None

        self._dLTdu_  = None  # @_on_L_gradients_()
        self._dLTduy_ = None  # .
        self._dLTduu_ = None  # .
        self._dLTdue_ = None  # .
        self._dLTdP_  = None  # .
        self._dLTdPP_ = None  # .
        self._dLTdH_  = None  # .
        self._dLTdHH_ = None  # .
        self._dLPdP_  = None  # .
        self._dLPdPP_ = None  # .
        self._dLHdH_  = None  # .
        self._dLHdHH_ = None  # .

        self._Wu_     = None  # .
        self._WP_     = None  # .
        self._WH_     = None  # .

        self._dWudP_  = None  # @_on_Wu_gradients_()
        self._dWudPP_ = None  # .
        self._dWudH_  = None  # .
        self._dWudHH_ = None  # . (assumed = 0)
        self._dWPdu_  = None  # @_on_WP_gradients_()
        self._dWPduu_ = None  # .
        self._dWPdH_  = None  # .
        self._dWPdHH_ = None  # . (assumed = 0)
        self._dWHdu_  = None
        self._dWHduu_ = None
        self._dWHdP_  = None
        self._dWHdPP_ = None

        self._dVTdu_      = None
        self._dVTduu_     = None
        self._dVTduy_     = None
        self._dVTdue_     = None
        self._SUM_dVPdP_  = None
        self._SUM_dVPdPP_ = None
        self._SUM_dVHdH_  = None
        self._SUM_dVHdHH_ = None

        self._SUM_LT_     = sym_shasf("int[L(t)]dt")
        self._SUM_Hu_     = sym_shasf("int[H(u,t)]dt")
        # self._SUM_Cu_     = sym_shamf("int[inv(dL(t)/duu)]dt")

        self._SUM_dLTduu_ = sym_shamf("int[dL(t)/duu]dt")

        self._SUM_dLTdP_  = sym_shavf("int[dL(t)/dP ]dt")
        self._SUM_dLTdPP_ = sym_shamf("int[dL(t)/dPP]dt")

        self._SUM_dLTdH_  = sym_shavf("int[dL(t)/dH ]dt")
        self._SUM_dLTdHH_ = sym_shamf("int[dL(t)/dHH]dt")

        self._SUM_dWudP_  = sym_shavf("int[dWu(P)/dP ]dt")
        self._SUM_dWudPP_ = sym_shamf("int[dWu(P)/dPP]dt")

        self._SUM_dWudH_  = sym_shavf("int[dWu(H)/dH ]dt")
        self._SUM_dWudHH_ = sym_shamf("int[dWu(H)/dHH]dt")

        self._SUM_dWPdH_  = sym_shavf("int[dWP(H)/dH ]dt")
        self._SUM_dWPdHH_ = sym_shamf("int[dWP(H)/dHH]dt")

        # self._SUM_dWHdP_  = sym_shamf("int[dWH(P)/dP ]dt")
        # self._SUM_dWHdPP_ = sym_shamf("int[dWP(P)/dPP]dt")

        # derivative operators for y, u, e
        self._DOy_ = sym_shamf("Derivative operator [y]")
        self._DOu_ = sym_shamf("Derivative operator [u]")
        self._DOe_ = sym_shamf("Derivative operator [e]")

        self._vfe_ = sym_shasf("Variational free-energy")

        self.InputVariables = ()
        self._collect_args_()

    def initialise(self, linker='cvm', optimizer='fast_run'):
        """Initialise DEM method.

        Dynamic Expectation Maximisation (DEM) ...
        """
        self._on_gradients_()
        self._on_LAP_()
        self._on_jacobian_()
        self._on_aposteriori_tdep_()
        self._on_aposteriori_tind_()
        self._on_aposteriori_ccov_()
        self._on_entropies_()

        compilemode = theano.compile.mode.Mode(linker=linker, optimizer=optimizer)
        self._compile_updates_tdep_(compilemode)
        self._compile_updates_sums_(compilemode)
        self._compile_updates_tind_(compilemode)
        self._compile_slicer_(compilemode)
        self._compile_misc_(compilemode)

        self.reset()
        self.set_qlp()
        self.print_args()

    def reset(self):
        """Initialise variables, integrals, and derivative operators.

        (...)
        """
        EOX = np.cast['int32'](self.hdm.EMBED_ORDER_X)
        EOV = np.cast['int32'](self.hdm.EMBED_ORDER_V)

        # initialise data, state variables with zeros
        self.hdm.y.set_value(
            np.zeros(
                (self._num_ydim_embed_,),
                dtype=theano.config.floatX))

        xN = self._num_xdim_embed_
        vN = self._num_vdim_embed_
        self.hdm.u.set_value(
            np.zeros(
                (xN + vN,),
                dtype=theano.config.floatX))

        # initialise prior expectation of ROOT causal variable with zeros
        eN = self._num_edim_embed_
        self.hdm.e.set_value(
            np.zeros(
                (eN,),
                dtype=theano.config.floatX))

        # initialise parameter its prior expectation with zeros
        pN = self._num_Pdim_
        self.hdm.P.set_value(
            np.zeros(
                (pN,),
                dtype=theano.config.floatX))
        self.hdm.p.set_value(
            np.zeros(
                (pN,),
                dtype=theano.config.floatX))

        # initialise hyperparameter its prior expectation with zeros
        hN = self._num_Hdim_
        self.hdm.H.set_value(
            np.zeros(
                (hN,),
                dtype=theano.config.floatX))
        self.hdm.h.set_value(
            np.zeros(
                (hN,),
                dtype=theano.config.floatX))

        # initalise derivative operators
        _IX_ = np.eye(EOX, k=1)
        _IV_ = np.eye(EOV, k=1)
        _dy_ = scipy.linalg.kron(_IX_, np.eye(self._num_ydim_))
        diag = list(itertools.chain(
            *map(lambda x, v: [scipy.linalg.kron(_IX_, np.eye(x)),
                               scipy.linalg.kron(_IV_, np.eye(v))],
                 self.hdm.xdim,
                 self.hdm.vdim)))
        # for n in range(N):
        #     _dx_ = scipy.linalg.kron(_IX_, np.eye(self.hdm.xdim[n]))
        #     _dv_ = scipy.linalg.kron(_IV_, np.eye(self.hdm.vdim[n]))
        #     diag.append(_dx_)
        #     diag.append(_dv_)
        _du_ = scipy.linalg.block_diag(*diag)
        _de_ = scipy.linalg.kron(_IV_, np.eye(self._num_edim_))

        self._DOy_.set_value(_dy_)
        self._DOu_.set_value(_du_)
        self._DOe_.set_value(_de_)

        # initialise integrals
        self._reset_integrals_()

    def set_qlp(self,
                lpx=-1, lpv=-1, lpP=-1, lpH=-1,
                xrough=0.5, vrough=0.5):
        """Set/reset log-precision of emsemble densities.

        Log-precisions and covariance matrices for ensemble densities
        covariance matrices for q(u) will be embedded using default roughness. These quantities are updated by conditional estimates.
        """
        N = self.hdm.RANK
        EOV = self.hdm.EMBED_ORDER_V
        EOX = self.hdm.EMBED_ORDER_X

        try:
            assert len(lpx) == N, (
                "Expect log-precisions for q(x) to have "
                "len={} but get {}.".format(N, len(lpx)))
        except TypeError:
            lpx = (lpx,) * N

        try:
            assert len(lpv) == N, (
                "Expect log-precisions for q(v) to have "
                "len={} but get {}.".format(N, len(lpv)))
        except TypeError:
            lpv = (lpv,) * N

        try:
            assert len(lpP) == N, (
                "Expect log-precisions for q(P) to have "
                "len={} but get {}.".format(N, len(lpP)))
        except TypeError:
            lpP = (lpP,) * N

        try:
            assert len(lpH) == N, (
                "Expect log-precisions for q(H) to have "
                "len={} but get {}.".format(N, len(lpH)))
        except TypeError:
            lpH = (lpH,) * N

        try:
            Cx = [np.eye(x) / np.exp(p)
                  for x, p in zip(self.hdm.xdim, lpx)]
            Cv = [np.eye(v) / np.exp(p)
                  for v, p in zip(self.hdm.vdim, lpv)]
            CP = [np.eye(P) / np.exp(p)
                  for P, p in zip(self.hdm.Pdim, lpP)]
            CH = [np.eye(H) / np.exp(p)
                  for H, p in zip(self.hdm.Hdim, lpH)]

        except AttributeError:
            raise AttributeError(
                "Use method `set_sizes()` to assign "
                "dimensions and sizes first!")

        # embed Cx and Cv
        emCx = [num_cemb(cx, EOX, xrough) for cx in Cx]
        emCv = [num_cemb(cv, EOV, vrough) for cv in Cv]
        emCu = list(itertools.chain(*zip(emCx, emCv)))

        self.Cu.set_value(
            scipy.linalg.block_diag(*emCu))
        self.CP.set_value(
            scipy.linalg.block_diag(*CP))
        self.CH.set_value(
            scipy.linalg.block_diag(*CH))

        self._num_lpx_ = lpx
        self._num_lpv_ = lpv
        self._num_lpP_ = lpP
        self._num_lpH_ = lpH

    def set_params(self, P):
        """Set parameters (optional)."""
        assert len(P) == self._num_Pdim_
        self.hdm.P.set_value(P)

    def set_hyparams(self, H):
        """Set hyperparameters (optional)."""
        assert len(H) == self._num_Hdim_
        self.hdm.H.set_value(H)

    def set_state_prior(self, E, rough=0.5):
        """Set prior expectation on (top-level) causal state (followed by embedding)."""
        if np.ndim(E) == 1:
            E = np.reshape(E, (1, -1))
        E = num_temb(E, self.hdm.EMBED_ORDER_V, rough)
        self.hdm.e.set_value(E)

    def set_dt(self, dt):
        """Set LM step size (optional, default=1)."""
        self._dt_.set_value(dt)

    def set_extras(self, *extras):
        """Set `extra` model arguments."""
        nex = len(extras)
        if nex == 0:
            print(
                ("[{}] Model expects Extra arguments "
                 "in following order:".format(datetime.datetime.now())))
            self.print_args()
        else:
            assert nex % 2 == 0
            for i in range(0, nex, 2):
                self.InputVariables[extras[i]].set_value(extras[i + 1])

    def set_data(self, Y):
        """Set observation (without embedding)."""
        tY, dY = np.shape(Y)
        if hasattr(self, '_num_ydim_'):
            assert self._num_ydim_ == dY
        self.Y = np.asarray(Y)
        self.tY = tY

        self.CEu = np.zeros((tY, self._num_udim_embed_))
        self.CEP = np.zeros((self._num_Pdim_,))
        self.CEH = np.zeros((self._num_Hdim_,))

    def invert(self,
               DEM=(1, 1, 1),
               dt=1,
               t0=0,
               TOL=np.exp(-4),
               msg=True):
        """Model inversion."""
        maxD, maxE, maxM = DEM
        nE, nD, nM = 0, 0, 0
        EOX = self.hdm.EMBED_ORDER_X

        for nE in range(maxE):
            self._reset_integrals_()
            for t in range(self.tY):
                for nD in range(maxD):
                    y = num_temb(self.Y, EOX, t + nD / maxD)
                    self._set_data_(y)
                    u0, tolu, du = self._Fcn_APu_TR_()
                    if tolu < TOL:
                        break
                self.CEu[t, :] = u0 + du
                self._Fcn_UpdateIntegrals_()

            for nM in range(maxM):
                H0, tolH, dH = self._Fcn_APH_()
                if tolH < TOL:
                    break
            self.CEH[:] = H0 + dH

            P0, tolP, dP = self._Fcn_APP_()
            self._msg_invert_ = (
                "DEM [{:2d}:{:2d}:{:2d}] "
                "F={:.4f}; "
                "|∆u|={:.4f}; "
                "|∆P|={:.4f}; "
                "|∆H|={:.4f}"
                "\n").format(nE, nD, nM,
                             np.asscalar(self._Fcn_VFE_()),
                             np.asscalar(tolu),
                             np.asscalar(tolP),
                             np.asscalar(tolH))
            if msg:
                print(self._msg_invert_, flush=True)
            if tolP < TOL:
                break
        self.CEP[:] = P0 + dP

    def integrate(self, t):
        """Integrate system to a specific time."""
        pass

    def print_args(self, which=None):
        """Print `extra` arguments."""
        if which is None:
            print((
                "Use method `set_extras()` to set the following "
                "variables. (if applicable)"))
            for i, a in enumerate(self.InputVariables):
                print('{:3d} = {}'.format(i, a.name))
        elif type(which) is int:
            pass
        elif type(which) in (list, tuple):
            pass

    def _set_data_(self, num_y_embedded):
        N = len(num_y_embedded)
        assert N % self._num_ydim_ == 0, (
            "Illegal data size: expect {} "
            "but {} is given.".format(self._num_ydim * self.hdm.EMBED_ORDER_X, N))
        self.hdm.y.set_value(num_y_embedded)

    def _on_gradients_(self):
        self._on_err_gradients_()
        self._on_L_gradients_()
        self._on_Wu_gradients_()
        self._on_WP_gradients_()
        self._on_WH_gradients_()

    def _on_err_gradients_(self):
        xerr = self.hdm.xerror  # state error
        verr = self.hdm.verror  # cause error
        E = [sym_jac(xei, self.hdm.P) for xei in xerr]
        B = [sym_jac(vei, self.hdm.P) for vei in verr]

        self._dEdP_ = E
        self._dBdP_ = B

        E = [sym_jac(xei, self.hdm.u) for xei in xerr]
        B = [sym_jac(vei, self.hdm.u) for vei in verr]

        self._dEdu_ = E
        self._dBdu_ = B

        E = [sym_jac(xei, self.hdm.y) for xei in xerr]
        B = [sym_jac(vei, self.hdm.y) for vei in verr]

        self._dEdy_ = E
        self._dBdy_ = B

        E = [sym_jac(xei, self.hdm.e) for xei in xerr]
        B = [sym_jac(vei, self.hdm.e) for vei in verr]

        self._dEde_ = E
        self._dBde_ = B

    def _on_L_gradients_(self):
        # unpack variables
        Ax = self.hdm.xinfo
        Av = self.hdm.vinfo

        E = self.hdm.xerror
        B = self.hdm.verror

        dEdu = self._dEdu_
        dBdu = self._dBdu_
        dEdy = self._dEdy_
        dBdy = self._dBdy_
        dEdP = self._dEdP_
        dBdP = self._dBdP_
        dEde = self._dEde_
        dBde = self._dBde_

        print("[{}] A.D. over L(t) wrt [y, u, e].".format(datetime.datetime.now()))
        # self._dLTdu_  = sym_jac(self.LT, self.hdm.u)
        # self._dLTduy_ = sym_jac(self._dLTdu_, self.hdm.y)
        # self._dLTduu_ = sym_hes(self.LT, self.hdm.u)
        # self._dLTdue_ = sym_jac(self._dLTdu_, self.hdm.e)

        # [dLT/du]
        # ----- ----- ----- ----- ----- ----- ----- -----
        dLTdu = [
            - dedu.T.dot(ax).dot(e) - dbdu.T.dot(av).dot(b)
            for dedu, ax, e, dbdu, av, b
            in zip(dEdu, Ax, E, dBdu, Av, B)]
        self._dLTdu_ = reduce_sum(dLTdu)

        # [dLT/dudy]
        # ----- ----- ----- ----- ----- ----- ----- -----
        dLTduy = [
            - dedu.T.dot(ax).dot(dedy) - dbdu.T.dot(av).dot(dbdy)
            for dedu, ax, dedy, dbdu, av, dbdy
            in zip(dEdu, Ax, dEdy, dBdu, Av, dBdy)]
        self._dLTduy_ = reduce_sum(dLTduy)

        # [dLT/dudu]
        # ----- ----- ----- ----- ----- ----- ----- -----
        dLTduu = [
            - dedu.T.dot(ax).dot(dedu) - dbdu.T.dot(av).dot(dbdu)
            for dedu, ax, dbdu, av
            in zip(dEdu, Ax, dBdu, Av)]
        self._dLTduu_ = reduce_sum(dLTduu)

        # [dLT/dude]
        # ----- ----- ----- ----- ----- ----- ----- -----
        dLTdue = [
            - dedu.T.dot(ax).dot(dede) - dbdu.T.dot(av).dot(dbde)
            for dedu, ax, dede, dbdu, av, dbde
            in zip(dEdu, Ax, dEde, dBdu, Av, dBde)]
        self._dLTdue_ = reduce_sum(dLTdue)

        print("[{}] A.D. over L(t) wrt [P].".format(datetime.datetime.now()))
        # [dLT/dP]
        # ----- ----- ----- ----- ----- ----- ----- -----
        dLTdP = [
            - dedp.T.dot(ax).dot(e) - dbdp.T.dot(av).dot(b)
            for dedp, ax, e, dbdp, av, b
            in zip(dEdP, Ax, E, dBdP, Av, B)]
        self._dLTdP_ = reduce_sum(dLTdP)

        print("[{}] A.D. over L(t) wrt [PP].".format(datetime.datetime.now()))
        # [dLT/dPdP]
        # ----- ----- ----- ----- ----- ----- ----- -----
        dLTdPP = [
            - dedp.T.dot(ax).dot(dedp) - dbdp.T.dot(av).dot(dbdp)
            for dedp, ax, dbdp, av
            in zip(dEdP, Ax, dBdP, Av)]
        self._dLTdPP_ = reduce_sum(dLTdPP)

        print("[{}] A.D. over L(t) wrt [H].".format(datetime.datetime.now()))
        # [dLT/dH]
        # ----- ----- ----- ----- ----- ----- ----- -----
        # 1. ee' = E
        # 2. H -> A
        # 3. LT = -0.5 * e'Ae + 0.5 * lndet(A)
        # 4. dLTdH = -0.5 * (E: - inv(P):)'dPdH:

        EE = [
            e.T.dot(e)
            for e in
            map(lambda x: x.dimshuffle([0, 'x']), E)]

        BB = [
            b.T.dot(b)
            for b in
            map(lambda x: x.dimshuffle([0, 'x']), B)]

        iAx = [sym_inv(ax) for ax in Ax]
        iAv = [sym_inv(av) for av in Av]

        dAxdH = [sym_jac(ax.flatten(1), self.hdm.H) for ax in Ax]
        dAvdH = [sym_jac(av.flatten(1), self.hdm.H) for av in Av]

        _dLTdH_ = reduce_sum(
            [-0.5 * (ee - iax).flatten(1).dot(daxdh)
             for ee, iax, daxdh
             in zip(EE, iAx, dAxdH)])
        _dLTdH_ = _dLTdH_ + reduce_sum(
            [-0.5 * (bb - iav).flatten(1).dot(davdh)
             for bb, iav, davdh
             in zip(BB, iAv, dAvdH)])

        print("[{}] A.D. over L(t) wrt [HH].".format(datetime.datetime.now()))
        # [dLT/dHdH]
        # ----- ----- ----- ----- ----- ----- ----- -----
        _dLTdHH_ = sym_jac(_dLTdH_, self.hdm.H)
        self._dLTdH_  = sym_jac(self.LT, self.hdm.H)
        self._dLTdHH_ = _dLTdHH_

        print("[{}] A.D. over L(P) wrt [P/PP].".format(datetime.datetime.now()))
        # [dLP/dP] & [dLP/dPdP]
        # ----- ----- ----- ----- ----- ----- ----- -----
        self._dLPdP_  = sym_jac(self.LP, self.hdm.P)
        self._dLPdPP_ = sym_hes(self.LP, self.hdm.P)

        print("[{}] A.D. over L(H) wrt [H/HH].".format(datetime.datetime.now()))
        # [dLH/dH] & [dLH/dHdH]
        # ----- ----- ----- ----- ----- ----- ----- -----
        self._dLHdH_  = sym_jac(self.LH, self.hdm.H)
        self._dLHdHH_ = sym_hes(self.LH, self.hdm.H)

        print((
            "[{}] Symbolic variables "
            "Wu, WP, WH.".format(datetime.datetime.now())))
        self._Wu_ = sym_tr(self.Cu.dot(self._dLTduu_)) / 2
        self._WP_ = sym_tr(self.CP.dot(self._dLTdPP_)) / 2
        self._WH_ = sym_tr(self.CH.dot(self._dLTdHH_)) / 2

    def _on_Wu_gradients_(self):
        Wu = self._Wu_

        # [dWu/dP]
        # ----- ----- ----- ----- ----- ----- ----- -----
        print("[{}] A.D. over Wu wrt [P].".format(datetime.datetime.now()))
        self._dWudP_ = sym_jac(Wu, self.hdm.P)

        # [dWu/dPdP]
        # ----- ----- ----- ----- ----- ----- ----- -----
        print("[{}] A.D. over Wu wrt [PP].".format(datetime.datetime.now()))
        self._dWudPP_ = sym_hes(Wu, self.hdm.P)

        # [dWu/dH]
        # ----- ----- ----- ----- ----- ----- ----- -----
        print("[{}] A.D. over Wu wrt [H].".format(datetime.datetime.now()))
        self._dWudH_ = sym_jac(Wu, self.hdm.H)

        # [dWu/dHdH]
        # ----- ----- ----- ----- ----- ----- ----- -----
        print("[{}] A.D. over Wu wrt [HH].".format(datetime.datetime.now()))
        self._dWudHH_ = sym_hes(Wu, self.hdm.H)

    def _on_WP_gradients_(self):
        # unpack variables
        WP = self._WP_

        # [dWP/du]
        # ----- ----- ----- ----- ----- ----- ----- -----
        print("[{}] A.D. over WP wrt [u].".format(datetime.datetime.now()))
        self._dWPdu_  = sym_jac(WP, self.hdm.u)

        # [dWP/dudu]
        # ----- ----- ----- ----- ----- ----- ----- -----
        print("[{}] A.D. over WP wrt [uu].".format(datetime.datetime.now()))
        self._dWPduu_ = sym_hes(WP, self.hdm.u)

        # [dWP/dH]
        # ----- ----- ----- ----- ----- ----- ----- -----
        print("[{}] A.D. over WP wrt [H].".format(datetime.datetime.now()))
        self._dWPdH_  = sym_jac(WP, self.hdm.H)

        # [dWP/dHdH]
        # ----- ----- ----- ----- ----- ----- ----- -----
        print("[{}] A.D. over WP wrt [HH].".format(datetime.datetime.now()))
        self._dWPdHH_ = sym_hes(WP, self.hdm.H)

    def _on_WH_gradients_(self):
        self._dWHdu_  = np.cast[theano.config.floatX](0)
        self._dWHduu_ = np.cast[theano.config.floatX](0)
        self._dWHdP_  = np.cast[theano.config.floatX](0)
        self._dWHdPP_ = np.cast[theano.config.floatX](0)

    def _on_LAP_(self):
        print("[{}] LAP and variational energies.".format(datetime.datetime.now()))
        # VT = LT + WP + (WH)
        #  => dVTdu  = dLTdu  + dWPdu   (+ dWHdu)
        #  => dVTduu = dLTduu + dWPduu  (+ dWHduu)
        self._dVTdu_  = self._dLTdu_  + self._dWPdu_
        self._dVTduu_ = self._dLTduu_ + self._dWPduu_
        self._dVTduy_ = self._dLTduy_
        self._dVTdue_ = self._dLTdue_

        # int[VP] = int[LT] + int[Wu] (+ int[WH]) + LP
        #  => int[dVPdP]  = int[dLTdP]  + int[dWudP]  (+ int[dWHdP])  + dLPdP
        #  => int[dVPdPP] = int[dLTdPP] + int[dWudPP] (+ int[dWHdPP]) + dLPdPP
        self._SUM_dVPdP_  = (
            self._SUM_dLTdP_  +
            self._SUM_dWudP_  +
            self._dLPdP_)
        self._SUM_dVPdPP_ = (
            self._SUM_dLTdPP_ +
            self._SUM_dWudPP_ +
            self._dLPdPP_)

        # int[VH] = int[LT] + int[Wu] + int[WP] + LH
        #  => int[dVHdH]  = int[dLTdH]  + int[dWudH]  + int[dWPdH]  + dLHdH
        #  => int[dVHdHH] = int[dLTdHH] + int[dWudHH] + int[dWPdHH] + dLHdHH
        self._SUM_dVHdH_  = (
            self._SUM_dLTdH_ +
            self._SUM_dWudH_ +
            self._SUM_dWPdH_ +
            self._dLHdH_)
        self._SUM_dVHdHH_ = (
            self._SUM_dLTdHH_ +
            self._SUM_dWudHH_ +
            self._SUM_dWPdHH_ +
            self._dLHdHH_)

        # --- --- --- --- --- --- ---
        # define motion of mode
        self._mmu_ = T.join(0, *[
            self._dVTdu_ + self._DOu_.dot(self.hdm.u),
            self._DOy_.dot(self.hdm.y),
            self._DOe_.dot(self.hdm.e)])

    def _on_jacobian_(self):
        print("[{}] Jacobian.".format(datetime.datetime.now()))
        Zyu = T.zeros((self.hdm.y.size, self.hdm.u.size))
        Zye = T.zeros((self.hdm.y.size, self.hdm.e.size))
        Zey = T.zeros((self.hdm.e.size, self.hdm.y.size))
        Zeu = T.zeros((self.hdm.e.size, self.hdm.u.size))

        J = [
            [self._dVTduu_ + self._DOu_, self._dVTduy_, self._dVTdue_],
            [Zyu,                        self._DOy_,    Zye          ],
            [Zeu,                        Zey,           self._DOe_   ]]

        J = [T.join(1, *Jr) for Jr in J]
        J = T.join(0, *J)

        self._J_ = J

    def _on_aposteriori_tdep_(self):
        print(("[{}] Updating scheme "
               "for conditional modes.".format(datetime.datetime.now())))
        Jdim = self._J_.shape[0]
        uN = self._num_udim_embed_

        _TRuD_ = (
            sym_expm_ss(self._dt_ * self._J_) -
            T.eye(Jdim)).dot(sym_pinv(self._J_)).dot(self._mmu_)
        _GNuD_ = - sym_pinv(self._J_).dot(self._mmu_)

        # ∆u
        # a posteriori u: u + ∆u -> u
        self._delta_APu_TR_ = _TRuD_[:uN]  # temporal regularisation (Levenberg-Marquadt)
        self._delta_APu_GN_ = _GNuD_[:uN]  # Gauss-Newton

    def _on_aposteriori_tind_(self):
        # ∆P
        # a posteriori P: P + ∆P -> P
        self._delta_APP_GN_ = - sym_pinv(self._SUM_dVPdPP_).dot(self._SUM_dVPdP_)

        # ∆H
        # a posteriori H: H + ∆H -> H
        self._delta_APH_GN_ = - sym_pinv(self._SUM_dVHdHH_).dot(self._SUM_dVHdH_)

    def _on_aposteriori_ccov_(self):
        print(("[{}] Updating scheme "
               "for conditional covariances.".format(datetime.datetime.now())))
        # conditional covariance of q(u), q(P) and q(H)
        self._Cu_ = - sym_inv(self._dLTduu_)
        self._CP_ = - sym_inv(self._dLPdPP_) - sym_pinv(self._SUM_dLTdPP_)
        self._CH_ = - sym_inv(self._dLHdHH_) - sym_pinv(self._SUM_dLTdHH_)

    def _on_entropies_(self):
        const = np.cast[theano.config.floatX](1 + np.log(2 * np.pi))
        self._Hu_ = T.cast(
            0.5 * sym_lndet(self.Cu) +
            0.5 * const * self._num_udim_embed_,
            theano.config.floatX)
        self._HP_ = (
            0.5 * sym_lndet(self.CP) +
            0.5 * const * self._num_Pdim_)
        self._HH_ = (
            0.5 * sym_lndet(self.CH) +
            0.5 * const * self._num_Hdim_)

    def _compile_updates_tdep_(self, compilemode):
        print(("[{}] Compiling conditional estimates "
               "for states.".format(datetime.datetime.now())))
        # - conditional mode and covariance on states
        #
        #
        #
        #
        self._Fcn_APu_GN_ = theano.function(
            [],
            [self.hdm.u,
             sym_norm(self._delta_APu_GN_, 1),
             self._delta_APu_GN_],
            updates=[
                (self.hdm.u, self.hdm.u + self._delta_APu_GN_),
                (self.Cu, self._Cu_)
            ],
            mode=compilemode)

        self._Fcn_APu_TR_ = theano.function(
            [],
            [self.hdm.u,
             sym_norm(self._delta_APu_TR_, 1),
             self._delta_APu_TR_],
            updates=[
                (self.hdm.u, self.hdm.u + self._delta_APu_TR_),
                (self.Cu, self._Cu_)],
            mode=compilemode)

    def _compile_updates_sums_(self, compilemode):
        # _Fcn_UpdateIntegrals_ is invoked when conditional
        # mode on state is optimised at given time. This evaluates
        # and accumulates derivatives necessary for inverting
        # parameters, hyperparameters and their conditional
        # covariances.
        self._Fcn_UpdateIntegrals_ = theano.function(
            [], [],
            updates=[
                (self._SUM_LT_,     self._SUM_LT_     + self.LT      ),
                (self._SUM_Hu_,     self._SUM_Hu_     + self._Hu_    ),
                # (self._SUM_Cu_,     self._SUM_Cu_     + self._Cu_    ),
                (self._SUM_dLTduu_, self._SUM_dLTduu_ + self._dLTduu_),
                (self._SUM_dLTdP_,  self._SUM_dLTdP_  + self._dLTdP_ ),
                (self._SUM_dLTdPP_, self._SUM_dLTdPP_ + self._dLTdPP_),
                (self._SUM_dLTdH_,  self._SUM_dLTdH_  + self._dLTdH_ ),
                (self._SUM_dLTdHH_, self._SUM_dLTdHH_ + self._dLTdHH_),
                (self._SUM_dWudP_,  self._SUM_dWudP_  + self._dWudP_ ),
                (self._SUM_dWudPP_, self._SUM_dWudPP_ + self._dWudPP_),
                (self._SUM_dWudH_,  self._SUM_dWudH_  + self._dWudH_ ),
                (self._SUM_dWPdH_,  self._SUM_dWPdH_  + self._dWPdH_ ),
                (self._SUM_dWudHH_, self._SUM_dWudHH_ + self._dWudHH_),
                (self._SUM_dWPdHH_, self._SUM_dWPdHH_ + self._dWPdHH_)
            ],
            mode=compilemode)

    def _compile_updates_tind_(self, compilemode):
        print(("[{}] Compiling conditional estimates "
               "for parameters.".format(datetime.datetime.now())))
        # - conditional mode and covariance on parameters (P)
        # return (1) P before update, (2) 1-norm of P displacement, and
        #        (3) P displacement.
        # update (1) P by displacement, and (2) conditional covariance on P
        # reset  (1) integrals associated with P
        self._Fcn_APP_ = theano.function(
            [],
            [self.hdm.P,
             sym_norm(self._delta_APP_GN_, 1),
             self._delta_APP_GN_],
            updates=[
                (self.hdm.P, self.hdm.P + self._delta_APP_GN_),
                (self.CP, self._CP_),
            ],
            mode=compilemode)

        print(("[{}] Compiling conditional estimates "
               "for hyperparameters.".format(datetime.datetime.now())))
        # - conditional mode and covariance on hyperparameters (H)
        #
        #
        #
        #
        self._Fcn_APH_ = theano.function(
            [],
            [self.hdm.H,
             sym_norm(self._delta_APH_GN_, 1),
             self._delta_APH_GN_],
            updates=[
                (self.hdm.H, self.hdm.H + self._delta_APH_GN_),
                (self.CH, self._CH_),
            ],
            mode=compilemode)

    def _compile_slicer_(self, compilemode):
        print(("[{}] Obtaining slicing info "
               "(y/x/v/P/H/e).".format(datetime.datetime.now())))
        self._Fcn_SlicingInfo_ = theano.function(
            [],
            [T.join(0, list(map(lambda a: T.join(0, a), self.hdm.yslice))),
             T.join(0, list(map(lambda a: T.join(0, a), self.hdm.xslice))),
             T.join(0, list(map(lambda a: T.join(0, a), self.hdm.vslice))),
             T.join(0, list(map(lambda a: T.join(0, a), self.hdm.Pslice))),
             T.join(0, list(map(lambda a: T.join(0, a), self.hdm.Hslice))),
             T.join(0, list(map(lambda a: T.join(0, a), self.hdm.eslice)))],
            on_unused_input='ignore',
            mode=compilemode)

    def _compile_misc_(self, compilemode):
        # --- --- --- --- --- --- ---
        # variational free energy
        self.vfe = (
            self._SUM_LT_ +
            self.LP +
            self.LH +
            0.5 * (
                sym_tr(self.Cu.dot(self._SUM_dLTduu_)) +
                sym_tr(self.CP.dot(self._SUM_dLTdPP_)) +
                sym_tr(self.CH.dot(self._SUM_dLTdHH_))) +
            (
                self._SUM_Hu_ +
                self._HP_ +
                self._HH_))

        print(("[{}] Finishing up."
               "".format(datetime.datetime.now())))
        self._Fcn_GetxPrediction_ = theano.function(
            [],
            self.hdm.xpredict,
            mode=compilemode)

        self._Fcn_GetvPrediction_ = theano.function(
            [],
            self.hdm.vpredict,
            mode=compilemode)

        self._Fcn_GetxError_ = theano.function(
            [],
            self.hdm.xerror,
            mode=compilemode)

        self._Fcn_GetvError_ = theano.function(
            [],
            self.hdm.verror,
            mode=compilemode)

        self._Fcn_VFE_ = theano.function(
            [],
            self.vfe,
            mode=compilemode)

    def _collect_args_(self):
        self.hdm._argExtra_.discard(None)
        self.InputVariables += tuple(self.hdm._argExtra_)

    def _reset_integrals_(self):
        uN = self._num_udim_embed_
        pN = self._num_Pdim_
        hN = self._num_Hdim_

        self._SUM_LT_.set_value(
            np.cast[theano.config.floatX](0))

        self._SUM_Hu_.set_value(
            np.cast[theano.config.floatX](0))

        # self._SUM_Cu_.set_value(np.zeros(
        #     (uN, uN),
        #     dtype=theano.config.floatX))

        self._SUM_dLTduu_.set_value(np.zeros(
            (uN, uN),
            dtype=theano.config.floatX))

        self._SUM_dLTdP_.set_value(np.zeros(
            (pN,),
            dtype=theano.config.floatX))

        self._SUM_dLTdPP_.set_value(np.zeros(
            (pN, pN),
            dtype=theano.config.floatX))

        self._SUM_dLTdH_.set_value(np.zeros(
            (hN,),
            dtype=theano.config.floatX))

        self._SUM_dLTdHH_.set_value(np.zeros(
            (hN, hN),
            dtype=theano.config.floatX))

        self._SUM_dWudP_.set_value(np.zeros(
            (pN,),
            dtype=theano.config.floatX))

        self._SUM_dWudPP_.set_value(np.zeros(
            (pN, pN),
            dtype=theano.config.floatX))

        self._SUM_dWudH_.set_value(np.zeros(
            (hN,),
            dtype=theano.config.floatX))

        self._SUM_dWPdH_.set_value(np.zeros(
            (hN,),
            dtype=theano.config.floatX))

        self._SUM_dWudHH_.set_value(np.zeros(
            (hN, hN),
            dtype=theano.config.floatX))

        self._SUM_dWPdHH_.set_value(np.zeros(
            (hN, hN),
            dtype=theano.config.floatX))


class ActiveDEM(DynamicExpectationMaximisation):
    def __init__(self, hdm):
        super().__init__(hdm)
        self.CEa = []
        EOV = self.hdm.EMBED_ORDER_V
        self._dEda_   = None
        self._dBda_   = None

        self._dLTda_  = None
        self._dLTdau_ = None
        self._dLTday_ = None
        self._dLTdae_ = None
        self._dLTdaa_ = None
        self._dLTdua_ = None

        self._dWPda_  = None
        self._dWPdua_ = None
        self._dWPdau_ = None
        self._dWPdaa_ = None

        self._dVTda_  = None
        self._dVTdua_ = None
        self._dVTdau_ = None
        self._dVTday_ = None
        self._dVTdae_ = None
        self._dVTdaa_ = None

        self._DOa_    = sym_shamf("Derivative operator [a]")

        self._num_adim_ = np.sum(self.hdm.adim).astype('int32')
        self._num_adim_embed_ = self._num_adim_ * EOV

    def reset(self):
        aN = self._num_adim_embed_
        self.hdm.a.set_value(
            np.zeros(
                (aN,),
                dtype=theano.config.floatX))

        _IA_ = np.eye(self.hdm.EMBED_ORDER_V, k=1)
        diag = list(
            map(lambda a: scipy.linalg.kron(_IA_, np.eye(a)),
                self.hdm.adim))
        _da_ = scipy.linalg.block_diag(*diag)
        self._DOa_.set_value(_da_)

        super().reset()

    def _on_err_gradients_(self):
        super()._on_err_gradients_()
        xerr = self.hdm.xerror
        verr = self.hdm.verror

        E = [sym_jac(xei, self.hdm.a) for xei in xerr]
        B = [sym_jac(vei, self.hdm.a) for vei in verr]

        self._dEda_ = E
        self._dBda_ = B

    def _on_L_gradients_(self):
        super()._on_L_gradients_()

        Ax = self.hdm.xinfo
        Av = self.hdm.vinfo

        E = self.hdm.xerror
        B = self.hdm.verror

        dEdu = self._dEdu_
        dBdu = self._dBdu_
        dEdy = self._dEdy_
        dBdy = self._dBdy_
        dEde = self._dEde_
        dBde = self._dBde_
        dEda = self._dEda_
        dBda = self._dBda_

        print("[{}] A.D. over L(t) wrt [a].".format(datetime.datetime.now()))
        # [dLT/da]
        # ----- ----- ----- ----- ----- ----- ----- -----
        dLTda = [
            - deda.T.dot(ax).dot(e) - dbda.T.dot(av).dot(b)
            for deda, ax, e, dbda, av, b
            in zip(dEda, Ax, E, dBda, Av, B)]
        self._dLTda_ = reduce_sum(dLTda)

        # [dLT/dau]
        # ----- ----- ----- ----- ----- ----- ----- -----
        dLTdau = [
            - deda.T.dot(ax).dot(dedu) - dbda.T.dot(av).dot(dbdu)
            for deda, ax, dedu, dbda, av, dbdu
            in zip(dEda, Ax, dEdu, dBda, Av, dBdu)]
        self._dLTdau_ = reduce_sum(dLTdau)

        # [dLT/day]
        # ----- ----- ----- ----- ----- ----- ----- -----
        dLTday = [
            - deda.T.dot(ax).dot(dedy) - dbda.T.dot(av).dot(dbdy)
            for deda, ax, dedy, dbda, av, dbdy
            in zip(dEda, Ax, dEdy, dBda, Av, dBdy)]
        self._dLTday_ = reduce_sum(dLTday)

        # [dLT/dae]
        # ----- ----- ----- ----- ----- ----- ----- -----
        dLTdae = [
            - deda.T.dot(ax).dot(dede) - dbda.T.dot(av).dot(dbde)
            for deda, ax, dede, dbda, av, dbde
            in zip(dEda, Ax, dEde, dBda, Av, dBde)]
        self._dLTdae_ = reduce_sum(dLTdae)

        # [dLT/daa]
        # ----- ----- ----- ----- ----- ----- ----- -----
        dLTdaa = [
            - deda.T.dot(ax).dot(deda) - dbda.T.dot(av).dot(dbda)
            for deda, ax, dbda, av
            in zip(dEda, Ax, dBda, Av)]
        self._dLTdaa_ = reduce_sum(dLTdaa)

        # [dLT/dua]
        # ----- ----- ----- ----- ----- ----- ----- -----
        dLTdua = [
            - dedu.T.dot(ax).dot(deda) - dbdu.T.dot(av).dot(dbda)
            for dedu, ax, deda, dbdu, av, dbda
            in zip(dEdu, Ax, dEda, dBdu, Av, dBda)]
        self._dLTdua_ = reduce_sum(dLTdua)

    def _on_WP_gradients_(self):
        super()._on_WP_gradients_()

        WP = self._WP_

        # [dWP/da]
        # ----- ----- ----- ----- ----- ----- ----- -----
        print("[{}] A.D. over WP wrt [a].".format(datetime.datetime.now()))
        self._dWPda_ = sym_jac(WP, self.hdm.a)

        # [dWP/duda]
        # ----- ----- ----- ----- ----- ----- ----- -----
        print("[{}] A.D. over WP wrt [ua].".format(datetime.datetime.now()))
        self._dWPdua_ = sym_jac(self._dWPdu_, self.hdm.a)

        # [dWP/dadu]
        # ----- ----- ----- ----- ----- ----- ----- -----
        print("[{}] A.D. over WP wrt [au].".format(datetime.datetime.now()))
        self._dWPdau_ = sym_jac(self._dWPda_, self.hdm.u)

        # [dWP/dada]
        # ----- ----- ----- ----- ----- ----- ----- -----
        print("[{}] A.D. over WP wrt [aa].".format(datetime.datetime.now()))
        self._dWPdaa_ = sym_hes(WP, self.hdm.a)

    def _on_LAP_(self):
        print("[{}] LAP and variational energies.".format(datetime.datetime.now()))
        # VT = LT + WP + (WH)
        #  => dVTdu  = dLTdu  + dWPdu   (+ dWHdu)
        #  => dVTduu = dLTduu + dWPduu  (+ dWHduu)
        self._dVTdu_  = self._dLTdu_  + self._dWPdu_
        self._dVTduu_ = self._dLTduu_ + self._dWPduu_
        self._dVTduy_ = self._dLTduy_
        self._dVTdue_ = self._dLTdue_
        self._dVTdua_ = self._dLTdua_ + self._dWPdua_

        self._dVTda_  = self._dLTda_  + self._dWPda_
        self._dVTdau_ = self._dLTdau_ + self._dWPdau_
        self._dVTday_ = self._dLTday_
        self._dVTdae_ = self._dLTdae_
        self._dVTdaa_ = self._dLTdaa_ + self._dWPdaa_

        # int[VP] = int[LT] + int[Wu] (+ int[WH]) + LP
        #  => int[dVPdP]  = int[dLTdP]  + int[dWudP]  (+ int[dWHdP])  + dLPdP
        #  => int[dVPdPP] = int[dLTdPP] + int[dWudPP] (+ int[dWHdPP]) + dLPdPP
        self._SUM_dVPdP_  = (
            self._SUM_dLTdP_  +
            self._SUM_dWudP_  +
            self._dLPdP_)
        self._SUM_dVPdPP_ = (
            self._SUM_dLTdPP_ +
            self._SUM_dWudPP_ +
            self._dLPdPP_)

        # int[VH] = int[LT] + int[Wu] + int[WP] + LH
        #  => int[dVHdH]  = int[dLTdH]  + int[dWudH]  + int[dWPdH]  + dLHdH
        #  => int[dVHdHH] = int[dLTdHH] + int[dWudHH] + int[dWPdHH] + dLHdHH
        self._SUM_dVHdH_  = (
            self._SUM_dLTdH_ +
            self._SUM_dWudH_ +
            self._SUM_dWPdH_ +
            self._dLHdH_)
        self._SUM_dVHdHH_ = (
            self._SUM_dLTdHH_ +
            self._SUM_dWudHH_ +
            self._SUM_dWPdHH_ +
            self._dLHdHH_)

        # --- --- --- --- --- --- ---
        # define motion of mode
        self._mmu_ = T.join(0, *[
            self._dVTdu_ + self._DOu_.dot(self.hdm.u),
            self._DOy_.dot(self.hdm.y),
            self._DOe_.dot(self.hdm.e),
            self._dVTda_ + self._DOa_.dot(self.hdm.a)])

    def _on_jacobian_(self):
        print("[{}] Jacobian.".format(datetime.datetime.now()))
        dVTduu = self._dVTduu_
        dVTduy = self._dVTduy_
        dVTdue = self._dVTdue_
        dVTdua = self._dVTdua_
        dVTdau = self._dVTdau_
        dVTday = self._dVTday_
        dVTdae = self._dVTdae_
        dVTdaa = self._dVTdaa_
        DOy = self._DOy_
        DOe = self._DOe_
        DOu = self._DOu_
        DOa = self._DOa_

        yN = self._num_ydim_embed_
        uN = self._num_udim_embed_
        eN = self._num_edim_embed_
        aN = self._num_adim_embed_

        Zyu = T.zeros((yN, uN))
        Zeu = T.zeros((eN, uN))
        Zey = T.zeros((eN, yN))
        Zye = T.zeros((yN, eN))
        Zya = T.zeros((yN, aN))
        Zea = T.zeros((eN, aN))

        J = [
            [dVTduu + DOu, dVTduy, dVTdue, dVTdua      ],
            [         Zyu,    DOy,    Zye,          Zya],
            [         Zeu,    Zey,    DOe,          Zea],
            [dVTdau      , dVTday, dVTdae, dVTdaa + DOa]]

        J = [T.join(1, *Jrow) for Jrow in J]
        J = T.join(0, *J)

        self._J_ = J

    def _on_aposteriori_tdep_(self):
        print(("[{}] Updating scheme "
               "for conditional modes.".format(datetime.datetime.now())))

        yN = self._num_ydim_embed_
        uN = self._num_udim_embed_
        eN = self._num_edim_embed_
        aN = self._num_adim_embed_

        dt = self._dt_
        J  = self._J_

        Jdim = yN + uN + eN + aN
        IJ = T.eye(Jdim)
        iJ = sym_pinv(J)
        eJ = sym_expm_ss(dt * J)
        U  = self._mmu_

        _TRuD_ = (eJ - IJ).dot(iJ).dot(U)
        _GNuD_ = - iJ.dot(U)

        # ∆u
        # a posteriori u: u + ∆u -> u
        self._delta_APu_TR_ = _TRuD_[:uN]
        self._delta_APu_GN_ = _GNuD_[:uN]

        # ∆a
        # a posteriori a: a + ∆a -> a
        self._delta_APa_TR_ = _TRuD_[-aN:]
        self._delta_APa_GN_ = _GNuD_[-aN:]

    def _compile_updates_tdep_(self, compilemode):
        print(("[{}] Compiling conditional estimates "
               "for states.".format(datetime.datetime.now())))
        # - conditional mode and covariance on states
        #
        #
        #
        #
        self._Fcn_APu_GN_ = theano.function(
            [],
            [self.hdm.u,
             sym_norm(self._delta_APu_GN_, 1),
             self._delta_APu_GN_,
             self.hdm.a,
             self._delta_APa_GN_],
            updates=[
                (self.hdm.u, self.hdm.u + self._delta_APu_GN_),
                (self.hdm.a, self.hdm.a + self._delta_APa_GN_),
                (self.Cu, self._Cu_)
            ],
            mode=compilemode)

        self._Fcn_APu_TR_ = theano.function(
            [],
            [self.hdm.u,
             sym_norm(self._delta_APu_TR_, 1),
             self._delta_APu_TR_,
             self.hdm.a,
            self._delta_APa_TR_],
            updates=[
                (self.hdm.u, self.hdm.u + self._delta_APu_TR_),
                (self.hdm.a, self.hdm.a + self._delta_APa_TR_),
                (self.Cu, self._Cu_)],
            mode=compilemode)
