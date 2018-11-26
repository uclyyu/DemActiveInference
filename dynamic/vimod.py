from visupport import *



class BaseModule:
    c = np.cast[theano.config.floatX](- 0.5 * np.log(2 * np.pi))

    def __init__(self):
        # initialise internal energies for
        # [0]: time-dependent states
        # [1]: parameters
        # [2]: hyperparameters
        self.L = [0., 0., 0.]
        self._extra_args_ = tuple()

    def on_energy(self):
        pass

    def add_energy(self, hdm):
        hdm.L[0] = hdm.L[0] + self.L[0]
        hdm.L[1] = hdm.L[1] + self.L[1]
        hdm.L[2] = hdm.L[2] + self.L[2]

    def declare_variable(self, _type_, _name_, *args):
        var = _type_(_name_, *args)
        self._extra_args_ += var,
        return var


# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #


# this will inject prior expectation (of v) to hdm top
class MasterModule(BaseModule):
    # dimension of state prior expectation e
    prior_dim = None

    def __init__(self, hdm, **opts):
        super().__init__()
        self.EOV = hdm.EMBED_ORDER_V

        self.request_state_variable(hdm, **opts)
        self.on_state_variable(**opts)
        self.register_state_variable(hdm)

        self.request_output(hdm)
        self.on_state_prediction()
        self.on_state_error()
        self.register_prediction(hdm)

        self.on_energy()
        self.add_energy(hdm)

    def request_state_variable(self, hdm, **opts):
        edim = self.prior_dim
        N = hdm.RANK - 1
        assert edim == hdm.vdim[N], (
            "improper Master module, "
            "dimension mismatch: "
            "dim(prior exptn)={} but"
            "dim(v)={}.".format(edim, hdm.vdim[N]))

        eslice, e, E = hdm.new_e(edim)
        self.eslice = eslice
        self.e = e
        self.E = E

    def on_state_variable(self, **opts):
        pass

    def register_state_variable(self, hdm):
        edim = self.prior_dim
        hdm.reg_e(edim, self.eslice, self.e, self.E, *self._extra_args_)

    def request_output(self, hdm):
        # use top causal state as output
        # this should have an order of EOV
        N = hdm.RANK - 1
        self.OUTPUT = hdm.Vc[N]

    def on_state_prediction(self):
        # simply use E[i] as prediction (prior expectation)
        # this can also be embedded time series (when using DEM)
        self.Epredict = [self.E[i] for i in range(self.EOV)]
        self.epredict = T.join(0, *self.Epredict)

    def on_state_error(self):
        self.Eerror = [self.OUTPUT[i] - self.Epredict[i] for i in range(self.EOV)]

        for i in range(self.EOV):
            self.Eerror[i].name = 'PREDICT ERR /e [m=M,d={}]'.format(i)

        self.eerror = T.join(0, *self.Eerror)
        self.eerror.name = 'PREDICT ERR /e [m=M]'

    def register_prediction(self, hdm):
        hdm.reg_vpe(self.epredict, self.eerror)


class GaussianMasterModule(MasterModule):
    def __init__(self, hdm, **opts):
        super().__init__(hdm, **opts)

    def on_state_variable(self, **opts):
        edim = self.prior_dim

        self.erough = self.declare_variable(
            sym_shasf,
            'ROUGHNESS   /e   [m=M; isca] (default=0.5)',
            0.5)
        self.eprec_val = self.declare_variable(
            sym_shasf,
            'PRECISION   /e   [m=M; isca] (log-default=2)',
            2)
        self.eprecision = sym_pemb(
            T.eye(edim) * T.exp(self.eprec_val),
            self.EOV,
            self.erough)

    def on_energy(self):
        edim = self.prior_dim

        self.L[0] = (
            -0.5   * self.eerror.dot(self.eprecision).dot(self.eerror) +
            0.5    * sym_lndet(self.eprecision) +
            np.cast[theano.config.floatX](self.c * edim * self.EOV))
        self.L[0].name = "L(t)[v/prior]"


class UnivariateGMModule(GaussianMasterModule):
    prior_dim = 1

# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #


class Module(BaseModule):
    # dimension of hidden (x) and causal (v) states
    states_dim = (None, None)
    # dimension of parameters (P) and hyperparameters (H)
    params_dim = (None, None)
    # dimension of connecting modules (v[m-1] or y)
    output_dim = None

    def __init__(self, hdm,
                 fG, fF, **opts):

        super().__init__()
        self.EOX = hdm.EMBED_ORDER_X
        self.EOV = hdm.EMBED_ORDER_V

        # self.xdim = xdim    # dimension of hidden state
        # self.vdim = vdim    # dimension of causal state
        # self.Psize = Pdim   # size/dimension of parameter
        # self.Hsize = Hdim   # size/dimension of hyperparameter

        self.xslice = None  # slice index of hdm.u
        self.x = None       # a slice of hdm.u according to xslice
        self.X = None       # x arranged into its motion in ascending order

        self.vslice = None  # slice index of hdm.u
        self.v = None       # a slice of hdm.u accourding to vslice
        self.V = None       # v arranged into its motion in ascending order

        self.Pslice = None  # slice index of hdm.P
        self.P = None       # a slice of hdm.P according to Pslice
        self.p = None       # a slice of hdm.p according to Pslice (prior expectation of P)

        self.Hslice = None  # slice index of hdm.H
        self.H = None       # a slice of hdm.H according to Hslice
        self.h = None       # a slice of hdm.h according to Hslice (prior expectation of H)

        self.initialise_module(hdm, fG, fF, **opts)

    def evaluate_state_equation(self, f, order=0):
        return f(self.X[order], self.V[order], self.P)

    def initialise_module(self, hdm, fG, fF, **opts):

        self.request_hdm_level(hdm, self.output_dim)

        self.check_equation(fG)
        self.check_equation(fF)

        self.request_output(hdm)
        self.on_output(**opts)

        self.request_state_variable(hdm, **opts)
        self.on_state_variable(**opts)
        self.register_state_variable(hdm)

        self.request_cause_variable(hdm, **opts)
        self.on_cause_variable(**opts)
        self.register_cause_variable(hdm)

        self.request_parameter(hdm, **opts)
        self.on_parameter(**opts)
        self.register_parameter(hdm)

        self.request_hyperparameter(hdm, **opts)
        self.on_hyperparameter(**opts)
        self.register_hyperparameter(hdm)

        self.on_state_prediction(fG, fF)
        self.on_state_error()
        self.register_state_info(hdm)

        self.on_param_error()
        self.on_hyparam_error()

        self.on_energy()  # ***
        self.add_energy(hdm)

    def request_hdm_level(self, hdm, odim):
        self.N = hdm._TOPUP_(odim)

    def request_output(self, hdm):
        self.OUTPUT = hdm.output

    def request_state_variable(self, hdm, **opts):
        # request or slice state variables from HDM
        # default behaviour is to make slices
        xdim = self.states_dim[0]
        xslice, x, X = hdm.new_x(xdim)
        self.xslice = xslice
        self.x = x
        self.X = X

    def on_state_variable(self, **opts):
        pass

    def register_state_variable(self, hdm):
        xdim = self.states_dim[0]
        hdm.reg_x(xdim, self.xslice, self.x, self.X)

    def request_cause_variable(self, hdm, **opts):
        vdim = self.states_dim[1]
        vslice, v, V = hdm.new_v(vdim)
        self.vslice = vslice
        self.v = v
        self.V = V

    def on_cause_variable(self, **opts):
        pass

    def register_cause_variable(self, hdm):
        vdim = self.states_dim[1]
        hdm.reg_v(vdim, self.vslice, self.v, self.V)

    def request_parameter(self, hdm, **opts):
        Pdim = self.params_dim[0]
        Pslice, P, p = hdm.new_P(Pdim)
        self.Pslice = Pslice
        self.P = P  # parameters (unknown)
        self.p = p  # prior expectation of parameters (given)

    def on_parameter(self, **opts):
        pass

    def register_parameter(self, hdm):
        Pdim = self.params_dim[0]
        hdm.reg_P(Pdim, self.P, self.p, self.Pslice, *self._extra_args_)

    def request_hyperparameter(self, hdm, **opts):
        Hdim = self.params_dim[1]
        Hslice, H, h = hdm.new_H(Hdim)
        self.Hslice = Hslice
        self.H = H  # hyperparameters (unknown)
        self.h = h  # prior expectation of hyperparameters (given)

    def on_hyperparameter(self, **opts):
        raise NotImplementedError

    def register_hyperparameter(self, hdm):
        Hdim = self.params_dim[1]
        hdm.reg_H(Hdim, self.H, self.h, self.Hslice, *self._extra_args_)

    def request_output(self, hdm):
        self.OUTPUT = hdm.output

    def on_output(self, **opts):
        pass

    def on_state_prediction(self, fG, fF):
        EOX = self.EOX
        EOV = self.EOV

        G = self.evaluate_state_equation(fG)
        F = self.evaluate_state_equation(fF)

        self.Vpredict = [
            T.Rop(G, self.X[0], self.X[i]) + T.Rop(G, self.V[0], self.V[i])
            if i < EOV else
            T.Rop(G, self.X[0], self.X[i])
            for i in range(EOX)]

        self.Xpredict = [
            T.Rop(F, self.X[0], self.X[i]) + T.Rop(F, self.V[0], self.V[i])
            if i < EOV else
            T.Rop(F, self.X[0], self.X[i])
            for i in range(EOX)]

        for i in range(EOX):
            self.Vpredict[i] = T.cast(self.Vpredict[i], theano.config.floatX)
            self.Vpredict[i].name = 'PREDICTION  /v   [m={},d={}]'.format(self.N, i)
        for i in range(EOX):
            self.Xpredict[i] = T.cast(self.Xpredict[i], theano.config.floatX)
            self.Xpredict[i].name = 'PREDICTION  /x   [m={},n={}]'.format(self.N, i)

        self.vpredict = T.join(0, *self.Vpredict)
        self.vpredict.name = 'PREDICTION  /v   [m={}]'.format(self.N)

        self.xpredict = T.join(0, *self.Xpredict)
        self.xpredict.name = 'PREDICTION  /x   [m={}]'.format(self.N)

    def on_state_error(self):
        EOX = self.EOX

        self.Verror = [self.OUTPUT[i] - self.Vpredict[i]
                       for i in range(EOX)]
        self.Xerror = [self.X[i + 1]  - self.Xpredict[i]
                       if i < EOX - 1 else
                       -self.Xpredict[i] for i in range(EOX)]

        for i in range(EOX):
            self.Verror[i].name = 'PREDICT ERR /v   [m={},d={}]'.format(self.N, i)
        for i in range(EOX):
            self.Xerror[i].name = 'PREDICT ERR /x   [m={},n={}]'.format(self.N, i)

        self.verror = T.join(0, *self.Verror)
        self.verror.name = 'PREDICT ERR /v   [m={}]'.format(self.N)

        self.xerror = T.join(0, *self.Xerror)
        self.xerror.name = 'PREDICT ERR /x   [m={}]'.format(self.N)

        # self.uerror = T.join(0, self.xerror, self.verror)
        # self.uerror.name = 'ERROR/u [m={}]'.format(self.N)

    def register_state_info(self, hdm):
        hdm.reg_xpe(self.xpredict, self.xerror)
        hdm.reg_vpe(self.vpredict, self.verror)

    def on_param_error(self):
        self.perror = self.P - self.p

    def on_hyparam_error(self):
        self.herror = self.H - self.h

    def set_head(self, hdm):
        hdm.__set_head(self.HEAD)

    def check_equation(self, eqn):
        pass



class ActiveModule(Module):
    states_dim = (None, None)
    params_dim = (None, None)
    action_dim = None
    output_dim = None

    def __init__(self, hdm, fG, fF, **opts):
        self.aslice = None
        self.a = None
        self.A = None
        super().__init__(hdm, fG, fF, **opts)

    def evaluate_state_equation(self, f, order=0):
        return f(self.A[order], self.X[order], self.V[order], self.P)

    def initialise_module(self, hdm, fG, fF, **opts):
        self.request_hdm_level(hdm, self.output_dim)

        self.check_equation(fG)
        self.check_equation(fF)

        self.request_output(hdm)
        self.on_output(**opts)

        self.request_state_variable(hdm, **opts)
        self.on_state_variable(**opts)
        self.register_state_variable(hdm)

        self.request_cause_variable(hdm, **opts)
        self.on_cause_variable(**opts)
        self.register_cause_variable(hdm)

        self.request_action_variable(hdm, **opts)
        self.on_action_variable(**opts)
        self.register_action_variable(hdm)

        self.request_parameter(hdm, **opts)
        self.on_parameter(**opts)
        self.register_parameter(hdm)

        self.request_hyperparameter(hdm, **opts)
        self.on_hyperparameter(**opts)
        self.register_hyperparameter(hdm)

        self.on_state_prediction(fG, fF)
        self.on_state_error()
        self.register_state_info(hdm)

        self.on_param_error()
        self.on_hyparam_error()

        self.on_energy()
        self.add_energy(hdm)

    def request_action_variable(self, hdm, **opts):
        adim = self.action_dim
        aslice, a, A = hdm.new_a(adim)
        self.aslice = aslice
        self.a = a
        self.A = A

    def on_action_variable(self, **opts):
        pass

    def register_action_variable(self, hdm):
        adim = self.action_dim
        hdm.reg_a(adim, self.aslice, self.a, self.A)


# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #


class GaussianModule(Module):


    def __init__(self, hdm, fG, fF, **opts):
        super().__init__(hdm, fG, fF, **opts)

    def on_parameter(self, **opts):
        Pdim = self.params_dim[0]
        # prior precision matrix using default log-precision
        self.pprec_val = self.declare_variable(
            sym_shasf,
            'PRIOR PRECSN/P [m={}; fsca] (log-default=1)'.format(self.N),
            1)
        self.pprecision = T.eye(Pdim) * T.exp(self.pprec_val)

    def on_hyperparameter(self, **opts):
        xdim, vdim = self.states_dim
        Pdim, Hdim = self.params_dim

        # prior precision matrix using default log-precision
        self.hprec_val = self.declare_variable(
            sym_shasf,
            'PRIOR PRECSN/H   [m={}; fsca] (log-default=1)'.format(self.N),
            1)
        self.hprecision = T.eye(Hdim) * T.exp(self.hprec_val)

        # declare precision components Qx & Qv
        # each column of Qx/Qv is a vectorised covariance component
        self.Qx = self.declare_variable(
            sym_shamf,
            'PRECN COMPNT/x   [m={}; fmat]'.format(self.N))

        self.Qv = self.declare_variable(
            sym_shamf,
            'PRECN COMPNT/v   [m={}; fmat]'.format(self.N - 1))

        nQx = self.Qx.shape[1]
        nQv = self.Qv.shape[1]

        Px = self.Qx * T.exp(self.H[:nQx])
        Px = Px.sum(axis=1).reshape((xdim, xdim))

        Pv = self.Qv * T.exp(self.H[nQx:nQx + nQv])
        Pv = Pv.sum(axis=1).reshape((vdim, vdim))

    def on_energy(self):
        xdim, vdim = self.states_dim
        Pdim, Hdim = self.params_dim
        EOX = len(self.X)
        EOV = len(self.OUTPUT)

        self.L[0] = \
            - 0.5 * (
                self.xerror.dot(self.xprecision).dot(self.xerror) +
                self.verror.dot(self.vprecision).dot(self.verror))\
            + 0.5 * (
                sym_lndet(self.xprecision) +
                sym_lndet(self.vprecision))\
            + self.c * (
                xdim * EOX + vdim * EOV)
        self.L[0].name += " + L(t)[u:{}]".format(self.N)

        self.L[1] = \
            - 0.5 * self.perror.dot(self.pprecision).dot(self.perror)\
            + 0.5 * sym_lndet(self.pprecision)\
            + self.c * Pdim
        self.L[1].name += " + L(P):{}".format(self.N)

        self.L[2] = \
            - 0.5 * self.herror.dot(self.hprecision).dot(self.herror)\
            + 0.5 * sym_lndet(self.hprecision)\
            + self.c * Hdim
        self.L[2].name += " + L(P):{}".format(self.N)


class GaussianActiveModule(ActiveModule):
    pass


class SimpleGaussianModule(Module):

    def __init__(self, hdm, fG, fF, **opts):
        # self.xprecision  precision matrix of state noise (x)
        # self.vprecision  precision matrix of state noise (v)
        # self.pprecision  prior precision over P
        # self.hprecision  prior precision over H

        super().__init__(hdm, fG, fF, **opts)

    def on_parameter(self, **opts):
        Pdim = self.params_dim[0]
        # prior precision matrix using default log-precision
        self.pprec_val = self.declare_variable(
            sym_shasf,
            'PRIOR PRECSN/P   [m={}; fsca] (log-default=1)'.format(self.N),
            1)
        self.pprecision = T.eye(Pdim) * T.exp(self.pprec_val)

    def on_hyperparameter(self, **opts):
        EOX = len(self.X)
        ydim = self.output_dim
        xdim = self.states_dim[0]
        Hdim = self.params_dim[1]
        # prior precision matrix using default log-precision
        self.hprec_val = self.declare_variable(
            sym_shasf,
            'PRIOR PRECSN/H   [m={}; fsca] (log-default=1)'.format(self.N),
            1)
        self.hprecision = T.eye(Hdim) * T.exp(self.hprec_val)

        # parameterise precision of state noise using hyperparameter,
        # followed by embedding

        # embedding roughness for state/observation noise
        self.xrough = self.declare_variable(
            sym_shasf,
            'ROUGHNESS   /x   [m={}; fsca] (default=0.5)'.format(self.N),
            0.5)
        self.vrough = self.declare_variable(
            sym_shasf,
            'ROUGHNESS   /v   [m={}; fsca] (default=0.5)'.format(self.N - 1),
            0.5)

        # use two values from hyperparameter as log-precisions
        # for x and v (y, if m=0), respectively.
        # plus fixed precision = 1
        Px = T.eye(xdim) + T.eye(xdim) * T.exp(self.H[0])
        Px.name = 'PRECISION   /x   [m={}] (of innovation w)'.format(self.N)
        Pv = T.eye(ydim) + T.eye(ydim) * T.exp(self.H[1])
        Pv.name = 'PRECISION   /v   [m={}] (of innovation z)'.format(self.N - 1)

        self.xprecision = sym_pemb(Px, EOX, self.xrough)
        self.xprecision.name = 'PRECISION   /x   [m={}] (of innovation w@gc)'.format(self.N)
        self.vprecision = sym_pemb(Pv, EOX, self.vrough)
        self.vprecision.name = 'PRECISION   /v   [m={}] (of innovation z@gc)'.format(self.N)

    def register_state_info(self, hdm):
        super().register_state_info(hdm)
        hdm.reg_xinfo(self.xprecision)
        hdm.reg_vinfo(self.vprecision)

    def on_energy(self):
        EOX = len(self.X)
        EOV = len(self.OUTPUT)
        xdim = self.states_dim[0]
        vdim = self.states_dim[1]
        Pdim = self.params_dim[0]
        Hdim = self.params_dim[1]

        self.L[0] = \
            - 0.5 * (
                self.xerror.dot(self.xprecision).dot(self.xerror) +
                self.verror.dot(self.vprecision).dot(self.verror))\
            + 0.5 * (
                sym_lndet(self.xprecision) +
                sym_lndet(self.vprecision))\
            + np.cast[theano.config.floatX](self.c * (
                xdim * EOX + vdim * EOV))
        self.L[0].name = 'L(t)[u:{}]'.format(self.N)

        self.L[1] = \
            - 0.5 * self.perror.dot(self.pprecision).dot(self.perror)\
            + 0.5 * sym_lndet(self.pprecision)\
            + self.c * Pdim
        self.L[1].name = 'L(P)'

        self.L[2] = \
            - 0.5 * self.herror.dot(self.hprecision).dot(self.herror)\
            + 0.5 * sym_lndet(self.hprecision)\
            + self.c * Hdim
        self.L[2].name = 'L(H)'


class SimpleGaussianActiveModule(ActiveModule, SimpleGaussianModule):
    def on_action_variable(self, **opts):
        adim = self.action_dim
        self.aprec_val = self.declare_variable(
            sym_shasf,
            'PRIOR PRECISION/a  [m={}; fsca] (log-default=0.1)'.format(self.N),
            0.1)
        self.aprecision = T.eye(adim * self.EOV) * T.exp(self.aprec_val)

    def on_energy(self):
        super().on_energy()
        adim = self.action_dim
        Lt_name = self.L[0].name
        self.L[0] = self.L[0] + (
            -0.5 * self.a.dot(self.aprecision).dot(self.a)
            +0.5 * sym_lndet(self.aprecision)
            + np.cast[theano.config.floatX](self.c * self.EOV * adim))
        self.L[0].name = Lt_name + " + L(t)[a/prior]"

# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #


class DemoLinearConvolutionModule(SimpleGaussianModule):
    states_dim = (2, 1)
    params_dim = (2, 2)
    output_dim = 4

    def get_g(self):
        # def _fG_(order, x, v, P):
        #     K0 = [[P[0],   .1633],
        #           [.1250,  .0676],
        #           [.1250, -.0676],
        #           [.1250, -.1633]]
        #     K0 = T.stack(K0)
        #     return sym_dot(K0, x[order])
        def _fG_(x, v, P):
            K0 = [[P[0],   .1633],
                  [.1250,  .0676],
                  [.1250, -.0676],
                  [.1250, -.1633]]
            K0 = T.stack(K0)
            return K0.dot(x)
        return _fG_

    def get_f(self):
        # def _fF_(order, x, v, P):
        #     nv = len(v)
        #
        #     K1 = [[-.25,  1.00],
        #           [P[1], - .25]]
        #     K1 = T.stack(K1)
        #     K2 = [[1.], [0.]]
        #     if order < nv:
        #         return sym_dot(K1, x[order]) + sym_dot(K2, v[order])
        #     else:
        #         return sym_dot(K1, x[order])
        def _fF_(x, v, P):
            K1 = [[-.25,  1.00],
                  [P[1], - .25]]
            K1 = T.stack(K1)
            K2 = T.stack([[1.], [0.]])
            return K1.dot(x) + K2.dot(v)
        return _fF_

    def __init__(self, hdm, **opts):
        super().__init__(hdm, self.get_g(), self.get_f(), **opts)


# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #

class DemoMountainCarModule(SimpleGaussianActiveModule):
    states_dim = (2, 1)
    params_dim = (7, 2)
    action_dim = 1
    output_dim = 2

    def get_g(self):
        def _fG_(a, x, v, P):
            return x
        return _fG_

    def get_f(self):
        def _fF_(a, x, v, P):
            p1 = P[0:1]
            p2 = P[1:3]
            p3 = P[3:5]
            p4 = P[5:7]
            pos = x[0:1]
            spe = x[1:2]

            r = 1 + 5 * (pos ** 2)

            b = ifelse(
                T.le(pos[0], 0),
                2 * pos + 1, (
                    r ** (-0.5) -
                    5 * (pos ** 2) * (r ** (-3 / 2)) -
                    (0.5 * pos) ** 4))
            c = p1 + p2.dot(x) + p3.dot(pos[0] * x) + p4.dot(spe[0] * x)

            dpos = spe
            dspe = -b - 0.25 * spe + v + T.tanh(a + c)
            return T.join(0, dpos, dspe)
        return _fF_

    def __init__(self, hdm, **opts):
        super().__init__(hdm, self.get_g(), self.get_f(), **opts)
