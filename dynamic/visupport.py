import collections
import functools
import theano
import theano.tensor as T
import theano.tensor.slinalg
import numpy as np
import scipy.linalg
import scipy.misc
from theano.ifelse import ifelse



class OrderedSet(collections.OrderedDict, collections.MutableSet):
    # source: http://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set
    def update(self, *args, **kwargs):
        if kwargs:
            raise TypeError("update() takes no keyword arguments")

        for s in args:
            for e in s:
                 self.add(e)

    def add(self, elem):
        self[elem] = None

    def discard(self, elem):
        self.pop(elem, None)

    def __le__(self, other):
        return all(e in other for e in self)

    def __lt__(self, other):
        return self <= other and self != other

    def __ge__(self, other):
        return all(e in self for e in other)

    def __gt__(self, other):
        return self >= other and self != other

    def __repr__(self):
        return 'OrderedSet([%s])' % (', '.join(map(repr, self.keys())))

    def __str__(self):
        return '{%s}' % (', '.join(map(repr, self.keys())))

    difference = property(lambda self: self.__sub__)
    difference_update = property(lambda self: self.__isub__)
    intersection = property(lambda self: self.__and__)
    intersection_update = property(lambda self: self.__iand__)
    issubset = property(lambda self: self.__le__)
    issuperset = property(lambda self: self.__ge__)
    symmetric_difference = property(lambda self: self.__xor__)
    symmetric_difference_update = property(lambda self: self.__ixor__)
    union = property(lambda self: self.__or__)

# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #


class BlockDiagOp(theano.Op):
    """Block diagonal."""

    def make_node(self, *arrays):
        """create an `Apply` node."""
        arrs = [theano.tensor.as_tensor_variable(ar) for ar in arrays]
        atyp = arrs[0].type()
        return theano.Apply(self, arrs, [atyp])

    def perform(self, node, inputs_storage, output_storage):
        """perform python implementation."""
        out = output_storage[0]

        bdiag = scipy.linalg.block_diag(*inputs_storage)
        out[0] = bdiag


class ExpmSSOp(theano.Op):
    """Matrix exponential using scaling and squaring."""

    # properties attributes
    __props__ = ()

    # `itypes` and `otypes` attributes are
    # compulsory if `make_node` method is not defined.
    # They are the type of input and output respectively
    # itypes = [theano.tensor.fmatrix]
    # otypes = [theano.tensor.fmatrix]

    # Compulsory if itypes and otypes are not defined
    def make_node(self, mat):
        """create an `Apply` node."""
        mat = theano.tensor.as_tensor_variable(mat)
        return theano.Apply(self, [mat], [mat.type()])

    def perform(self, node, inputs_storage, output_storage):
        """perform python implementation."""
        mat = inputs_storage[0]
        out = output_storage[0]

        cond = np.linalg.norm(mat, np.inf)
        powr = np.ceil(np.log2(cond)) + 1
        scale = 2 ** powr

        expm = scipy.linalg.expm(mat / scale)
        sqrm = np.linalg.matrix_power(
            expm,
            int(scale))

        out[0] = sqrm

    def infer_shape(self, node, input_shapes):
        return input_shapes


class ExpmSSEOp(theano.Op):
    """Matrix exponential using scaling and squaring the eigenvalues."""

    # properties attributes
    __props__ = ()

    # `itypes` and `otypes` attributes are
    # compulsory if `make_node` method is not defined.
    # They are the type of input and output respectively
    # itypes = [theano.tensor.fmatrix]
    # otypes = [theano.tensor.fmatrix]

    # Compulsory if itypes and otypes are not defined
    def make_node(self, mat):
        """create an `Apply` node."""
        mat = theano.tensor.as_tensor_variable(mat)
        return theano.Apply(self, [mat], [mat.type()])

    def perform(self, node, inputs_storage, output_storage):
        """perform python implementation."""
        mat = inputs_storage[0]
        out = output_storage[0]

        cond = np.linalg.norm(mat, np.inf)
        powr = np.ceil(np.log2(cond)) + 1
        scale = 2 ** powr

        expm = scipy.linalg.expm(mat / scale)

        # squaring the eigenvalues
        Ed, EV = np.linalg.eig(expm)
        iEV = np.linalg.inv(EV)
        Ed = np.diag(Ed ** scale)
        sqrm = EV.dot(Ed).dot(iEV)

        out[0] = sqrm

    def infer_shape(self, node, input_shapes):
        return input_shapes


# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #

def sym_shasi(name=None, value=0, **kwargs):
    # int64
    value = int(value)
    return theano.shared(value, name, strict=False, allow_downcast=True, **kwargs)

def sym_shasf(name=None, value=0, **kwargs):
    # float32
    value = np.cast[theano.config.floatX](value)
    return theano.shared(value, name, strict=False, allow_downcast=True, **kwargs)

def sym_shavf(name=None, **kwargs):
    value = np.zeros((0,), dtype=theano.config.floatX)
    return theano.shared(value, name, strict=False, allow_downcast=True, **kwargs)

def sym_shamf(name=None, **kwargs):
    value = np.zeros((0, 0), dtype=theano.config.floatX)
    return theano.shared(value, name, strict=False, allow_downcast=True, **kwargs)

def sym_shavi(name=None, **kwargs):
    value = np.zeros((0,), dtype=int)
    return theano.shared(value, name, strict=False, allow_downcast=True, **kwargs)

def sym_shami(name=None, **kwargs):
    value = np.zeros((0, 0), dtype=int)
    return theano.shared(value, name, strict=False, allow_downcast=True, **kwargs)

def sym_kron(A, B):
    return T.slinalg.kron(A, B)

def sym_inv(A):
    return T.nlinalg.matrix_inverse(A)

def sym_dot(A, B):
    return T.nlinalg.matrix_dot(A, B)

def sym_pinv(A):
    return T.nlinalg.pinv(A)

def num_expm(A):
    # numerical matrix exponential using Pade approximation
    # with scaling and squaring
    norm = np.linalg.norm(A, np.inf)
    n = np.ceil(np.log2(norm)) + 1
    s = 2 ** n
    P = scipy.linalg.expm(A / s)
    return np.linalg.matrix_power(P, int(s))


def sym_powm(M, s):
    """symbolic matrix power."""
    # TODO: implement a theano Op for this
    # (the one in theano is not fully symbolic)
    # the code below is very slow – large overhead for
    # a small code.
    d = M.shape[0]
    OUT, _ = theano.scan(
        fn=lambda prior_m, m: prior_m.dot(m),
        outputs_info=T.eye(d),
        non_sequences=M,
        n_steps=T.cast(s, 'int32'))
    return OUT[-1]


def sym_mnorm_inf(A):
    """infinity matrix norm, i.e., maximum absolute row sum."""
    M = T.abs_(A)
    return T.max(T.sum(M, axis=1))


# def sym_expm(A):
#     """expm using scaling and squaring."""
#     norm = sym_mnorm_inf(A)
#     n = T.ceil(T.log2(norm)) + 1
#     s = 2 ** n
#     P = T.slinalg.expm(A / s)
#
#     return sym_powm(P, s)

sym_expm_ss = ExpmSSOp()


# def sym_e_expm(A):
#     """expm using scaling and squaring eigenvalues."""
#     norm = sym_mnorm_inf(A)
#     n = T.ceil(T.log2(norm)) + 1
#     s = 2 ** n
#     P = T.slinalg.expm(A / s)
#     d, V = T.nlinalg.eig(P)
#     iV = T.nlinalg.matrix_inverse(V)
#     dp = d ** s
#
#     return V.dot(T.diag(dp)).dot(iV)

sym_expm_sse = ExpmSSEOp()

def sym_t_expm(A):
    """expm using Pade approximation (scipy routine)."""
    return T.slinalg.expm(A)

def sym_norm(A, n):
    return T.nlinalg.norm(A, n)

def sym_lndet(A):
    # NB this applies to positive def. matrices
    m = T.max(T.abs_(A))
    n = A.shape[0]
    Q = A / m
    detQ = T.nlinalg.det(Q)
    OUT = ifelse(
        T.eq(detQ, 0),
        n * T.log(m),
        n * T.log(m) + T.log(detQ))
    return T.cast(OUT, theano.config.floatX)

def sym_tr(A):
    return T.nlinalg.trace(A)

def reduce_sum(V):
    return functools.reduce(lambda x, y: x + y, V)

def sym_jac(*args, **kwargs):
    return T.jacobian(*args, disconnected_inputs='warn', **kwargs)

def sym_hes(*args, **kwargs):
    return T.hessian(*args, disconnected_inputs='warn', **kwargs)

# def sym_blkdiag(*arrays):
#     rsize = [A.shape[0] for A in arrays]
#     csize = [A.shape[1] for A in arrays]
#     Q = scipy.linalg.block_diag(*arrays)
#
#     r, c = Q.shape
#     for i in range(r):
#         for j in range(c):
#             if i == j: continue
#             Q[i, j] = T.zeros((rsize[i], csize[j]))
#     Q = Q.tolist()
#     R = []
#     for Qi in Q:
#         R.append(T.join(1, *Qi))
#     R = T.join(0, *R)
#     R.name = '_blkdiag_'
#     return R
sym_blkdiag = BlockDiagOp()

def sym_pemb(P, order, roughness):
    # embed precision matrix (symbolic operation)
    k = - T.ones(order)
    d = T.arange(order)
    k = k ** d
    d = d * 2
    x = roughness * T.sqrt(2)
    r = T.zeros(order)
    r = T.join(0, [T.cumprod(1 - d) / (x ** d), np.zeros(order)]).T
    r = r.flatten(1)[:-1]
    R = T.join(0, [r[i:i + order] for i in range(order)])
    R = R * k.reshape((1, order))
    R = T.nlinalg.matrix_inverse(R)
    return T.cast(sym_kron(R, P), theano.config.floatX)

def num_temb(X, order, te, dt=1, t0=0):
    # Embed time-series to its higher-order motion wrt time.
    # (numerical operation)
    # X : [N, d]-array
    #    d-dimensional 'observation' of N instances, whose
    #    first time bin marks the time `t0`
    # order : int
    #    embedding order
    # te : float
    X = np.asarray(X)

    N, d = X.shape
    bin_t, o = np.mgrid[0:order, 0:order]

    bin_e = np.fix((te - t0) / dt)
    bin_c = np.fix((order - 1) / 2)

    # local unembed operator
    E = (((bin_t - bin_c) * dt) ** o) / scipy.misc.factorial(o)
    # local embed operator
    try:
        T = np.linalg.inv(E)
    except np.linalg.LinAlgError as err:
        T = np.linalg.pinv(E)
        print("[t={:.3f}] Numpy LinAlgError: {}".format(te, err))

    # indices for X
    # boundary condition: padding with same instance
    idx = np.ogrid[0:order] - bin_c + bin_e
    idx[idx < 0] = 0
    idx[idx >= N] = N - 1
    idx = idx.astype(int)

    y =  scipy.linalg.kron(T, np.eye(d)).dot(X[idx, :].flatten())
    return y

def num_cemb(C, order, roughness):
    # temporal correlation matrix assuming Gaussian form
    k = - np.ones(order)
    d = np.ogrid[0:order]
    k = k ** d
    d = d * 2
    x = roughness * np.sqrt(2)
    r = np.zeros(order * 2 - 1)
    r[d] = np.cumprod(1 - d) / (x ** d)
    R = scipy.linalg.hankel(r[:order], r[order - 1:])
    R = R * k[None].T
    return scipy.linalg.kron(R, C)

# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #
