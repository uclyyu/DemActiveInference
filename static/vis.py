def vilaps(L, b, bdims, y=None, hp=None):
    """Variational inference under Laplace assumption."""
    # inputs:
    #  L - log generative density (symbolic, scalar)
    #  b – parameters (list of symbols, vector)
    #  bdims - dimension of individual parameters (list of int)
    #  y - observation (symbolic, 2D array)
    #  hp - hyperpriors (list of symbols, vector)
    # returns:
    #  
    
    def _jac_(*args, **kwargs):
        return T.jacobian(*args, disconnected_inputs='warn', **kwargs)
    
    def _hes_(*args, **kwargs):
        return T.hessian(*args, disconnected_inputs='warn', **kwargs)
    
    assert len(b) == len(bdims)
    
    if y is None:
        y = []
    else:
        y = [y]
       
    if hp is None:
        hp = []
    
    mfp = len(bdims)
    
    Li  = [_jac_(L, b[i]) for i in range(mfp)]
    Lii = [_hes_(L, b[i]).flatten(1) for i in range(mfp)]
       
    Liij  = [[T.join(0, [_jac_(Lii[i][k], b[j]) for k in range(bdims[i] ** 2)])
              for j in range(mfp)]
             for i in range(mfp)]
    Liijj = [[T.join(0, [_hes_(Lii[i][k], b[j]).flatten(1) for k in range(bdims[i] ** 2)])
              for j in range(mfp)]
             for i in range(mfp)]
    
    Ci  = [- T.nlinalg.pinv(Lii[i].reshape((bdims[i], bdims[i]))) for i in range(mfp)]
    fCi = [Ci[i].flatten(1) for i in range(mfp)]
    
    Uj = [[T.join(0, [T.sum(fCi[j] * Liij[j][i][:, k])  for k in range(bdims[i])])
           for j in range(mfp)
           if j != i]
          for i in range(mfp)]
    
    Ujj = [[T.join(0, [T.sum(fCi[j] * Liijj[j][i][:, k]) for k in range(bdims[i] ** 2)])
            for j in range(mfp)
            if j != i]
           for i in range(mfp)]
    
    Vi  = [Li[i] + 0.5 * functools.reduce(lambda x, y: x + y, Uj[i]) for i in range(mfp)]
    Vii = [(Lii[i] + 0.5 * functools.reduce(lambda x, y: x + y, Ujj[i])).reshape((bdims[i], bdims[i]))
           for i in range(mfp)]
    
    bGN = [-T.nlinalg.pinv(Vii[i]).dot(Vi[i]) for i in range(mfp)]
    bLM = [(T.slinalg.expm(Vii[i]) - 1).dot(T.nlinalg.pinv(Vii[i])).dot(Vi[i]) for i in range(mfp)]

    F = L
    for i in range(mfp):
        F += 0.5 * T.sum(fCi[i] * Lii[i] + T.log(T.nlinalg.det(Ci[i])))
        
    viupdate = theano.function(
        b + hp + y,
        bGN + bLM + Ci + [F],
        allow_input_downcast=True)
    
    return viupdate
