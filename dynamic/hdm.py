from visupport import *



class HierarchicalDynamicModel:
    def __init__(self, EOX=7, EOV=3):
        # EOX: int
        #    embedding order for hidden states (x)
        # EOV: int
        #    embedding order for causes (v)
        #
        # === attributes ===
        #    xslice: (2-tuple, [...])
        #        symbolic (or numeric?)

        assert EOX >= EOV

        self.RANK = 0
        self.EMBED_ORDER_X = int(EOX)
        self.EMBED_ORDER_V = int(EOV)
        self.HEAD = {'u': 0, 'P': 0, 'H': 0, 'e': 0}

        self.y = sym_shavf('DATA        /y   [m=0 ; fvec]@GC')
        self.u = sym_shavf('STATES      /x,v [m=1+; fvec]@GC')
        self.P = sym_shavf('PARAMETER   /P   [m=1+; fvec]')
        self.H = sym_shavf('HYPARAMETER /H   [m=1+; fvec]')
        self.e = sym_shavf('PRIOR EXPTN /e   [m=M ; fvec]@GC')
        self.p = sym_shavf('PRIOR EXPTN /P   [m=1+; fvec]')
        self.h = sym_shavf('PRIOR EXPTN /H   [m=1+; fvec]')

        self.ydim = ()
        self.xdim = ()
        self.vdim = ()
        self.edim = ()
        self.Pdim = ()
        self.Hdim = ()

        self.yslice = ((0, self.y.size),)

        self.xslice = tuple()
        self.xc = tuple()  # container for hierarchical state
        self.Xc = tuple()
        self.xpredict = tuple()
        self.xerror = tuple()
        self.xinfo = tuple()

        self.vslice = tuple()
        self.vc = tuple()
        self.Vc = tuple()
        self.vpredict = tuple()
        self.verror = tuple()
        self.vinfo = tuple()

        self.eslice = tuple()
        self.ec = tuple()
        self.Ec = tuple()

        self.Pslice = tuple()
        self.Pc = tuple()
        self.pc = tuple()

        self.Hslice = tuple()
        self.Hc = tuple()
        self.hc = tuple()

        self.output = None
        self.L = [0., 0., 0.]

        self._argExtra_ = OrderedSet()
        self._argExtra_.add(self.p)
        self._argExtra_.add(self.h)

    def new_e(self, edim):
        """Request a new slice of variable representing prior expectation over (top-level) causal state."""
        assert edim > 0 and type(edim) is int, (
            "requested state prior expectation e by invalid dimension {}"
            "".format(edim))
        self._NEW_OK_()

        N = self.RANK
        EOV = self.EMBED_ORDER_V

        TAIL = self.HEAD['e']
        HEAD = TAIL + edim * EOV
        eslice = (TAIL, HEAD)

        e = self.e[slice(*eslice)]
        e.name = 'PRIOR EXPTN /e [m={}; slice]'.format(N)

        E = tuple(e[slice(i * edim, (i + 1) * edim)] for i in range(EOV))
        for i in range(EOV):
            E[i].name = 'PRIOR EXPTN /e [m={},d={}; slice]'.format(N, i)

        return eslice, e, E

    def reg_e(self, dim, eslice, e, E, *extra):
        self.edim = dim,
        self.HEAD['e'] = eslice[1]
        self.eslice += eslice,
        self.ec += e,
        self.Ec += E,

        for ex in extra:
            self._argExtra_.add(ex)

    def new_x(self, xdim):
        assert xdim > 0 and type(xdim) is int, (
            "requested state x->u by invalid dimension {}"
            "".format(xdim))
        self._NEW_OK_()

        N = self.RANK
        EOX = self.EMBED_ORDER_X

        # slice index by orders of state motion
        TAIL = self.HEAD['u']
        HEAD = TAIL + xdim * EOX
        xslice = (TAIL, HEAD)

        x = self.u[slice(*xslice)]
        x.name = 'STATE       /x [m={}; slice]'.format(N)

        X = tuple(x[slice(i * xdim, (i + 1) * xdim)] for i in range(EOX))
        for i in range(EOX):
            X[i].name = 'STATE       /x [m={},n={}; slice]'.format(N, i)

        return xslice, x, X

    def qry_x(self, rank):
        pass

    def reg_x(self, dim, valid_slice, x, X):
        self.HEAD['u'] = valid_slice[1]
        self.xdim += dim,
        self.xslice += valid_slice,
        self.xc += x,
        self.Xc += X,

    def new_v(self, vdim):
        assert vdim > 0 and type(vdim) is int, (
            "requested state v->u by invalid dimension {}"
            "".format(vdim))
        self._NEW_OK_()

        N = self.RANK
        EOV = self.EMBED_ORDER_V

        # slice index by orders of state motion
        TAIL = self.HEAD['u']
        HEAD = TAIL + vdim * EOV
        vslice = (TAIL, HEAD)

        v = self.u[slice(*vslice)]
        v.name = 'STATE       /v [m={}; slice]'.format(N)

        V = tuple(v[slice(i * vdim, (i + 1) * vdim)] for i in range(EOV))
        for i in range(EOV):
            V[i].name = 'STATE       /v [m={},d={}; slice]'.format(N, i)

        return vslice, v, V

    def qry_v(self, rank):
        pass

    def reg_v(self, dim, valid_slice, v, V):
        self.HEAD['u'] = valid_slice[1]  # HEAD + vdim * EOV
        self.vdim += dim,
        self.vslice += valid_slice,
        self.vc += v,
        self.Vc += V,

    def reg_xpe(self, xp, xpe):
        self.xpredict += xp,
        self.xerror += xpe,

    def reg_vpe(self, vp, vpe):
        self.vpredict += vp,
        self.verror += vpe,

    def reg_xinfo(self, xinfo):
        self.xinfo += xinfo,

    def reg_vinfo(self, vinfo):
        self.vinfo += vinfo,

    def new_P(self, Pdim):
        assert Pdim > 0 and type(Pdim) is int, (
            "requested parameter P by invalid dimension {}"
            "".format(Pdim))
        self._NEW_OK_()

        N = self.RANK

        TAIL = self.HEAD['P']
        HEAD = TAIL + Pdim

        Pslice = (TAIL, HEAD)
        P = self.P[slice(*Pslice)]
        P.name = 'PARAM       /P [m={}; slice]'.format(N)

        p = self.p[slice(*Pslice)]
        p.name = 'PRIOR EXPTN /P [m={}; slice]'.format(N)

        return Pslice, P, p

    def qry_P(self):
        pass

    def reg_P(self, dim, P, p, Pslice, *extra):
        self.HEAD['P'] = Pslice[1]
        self.Pdim += dim,
        self.Pslice += Pslice,
        self.Pc += P,
        self.pc += p,

        for ex in extra:
            self._argExtra_.add(ex)

    def new_H(self, Hdim):
        assert Hdim > 0 and type(Hdim) is int, (
            "requested hyperparameter H by invalid dimension {}"
            "".format(Hdim))
        self._NEW_OK_()

        N = self.RANK

        TAIL = self.HEAD['H']
        HEAD = TAIL + Hdim

        Hslice = (TAIL, HEAD)
        H = self.H[slice(*Hslice)]
        H.name = 'HYPARAM     /H [m={}; slice]'.format(N)

        h = self.h[slice(*Hslice)]
        h.name = 'PRIOR EXPTN /H [m={}; slice]'.format(N)

        return Hslice, H, h

    def qry_H(self):
        pass

    def reg_H(self, dim, H, h, Hslice, *extra):
        self.HEAD['H'] += Hslice[1]
        self.Hdim += dim,
        self.Hc += H,
        self.hc += h,

        for ex in extra:
            self._argExtra_.add(ex)

    def _TOPUP_(self, dim):
        # caller should be instance of Module?
        self.RANK += 1
        if len(self.vdim) > 0:
            assert self.vdim[self.RANK - 1] == dim, (
                "dimension mismatch in causal state, "
                "HDM expects: {};"
                "Module expects: {}."
                "".format(self.vdim[self.RANK - 1], dim))
            self.output = self.Vs[self.RANK - 1]
        else:
            assert dim > 0 and type(dim) is int
            self.ydim += dim,
            self.output = self._unvecy_()
        return self.RANK

    def _SET_HEAD_(self, head):
        pass

    def _NEW_OK_(self):
        pass

    def _unvecy_(self):
        EOX = self.EMBED_ORDER_X
        ydim = self.ydim[0]
        return tuple(self.y[i * ydim:(i + 1) * ydim] for i in range(EOX))


class ActiveHDM(HierarchicalDynamicModel):
    """hierarchical dynamical model with action variable."""

    def __init__(self, EOX=3, EOV=2):
        super().__init__(EOX, EOV)
        self.a = sym_shavf('ACTION      /a   [m=1+; fvec]')
        self.aslice = ()
        self.adim = ()
        self.ac = ()
        self.Ac = ()
        self.HEAD['a'] = 0

    def new_a(self, adim):
        """doc string."""
        assert adim > 0 and type(adim) is int
        self._NEW_OK_()

        N = self.RANK
        EOV = self.EMBED_ORDER_V

        TAIL = self.HEAD['a']
        HEAD = TAIL + adim * EOV
        aslice = (TAIL, HEAD)

        a = self.a[slice(*aslice)]
        a.name = 'ACTION      /a   [m={:2d}; slice]'.format(N)

        A = tuple(a[slice(i * adim, (i + 1) * adim)] for i in range(EOV))
        for i in range(EOV):
            A[i].name = 'ACTION      /s   [m={},d={}; slice]'.format(N, i)

        return aslice, a, A

    def reg_a(self, adim, aslice, a, A):
        """doc string."""
        self.adim += adim,
        self.aslice += aslice,
        self.HEAD['a'] = aslice[1]
        self.ac += a,
        self.Ac += A,

# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #
