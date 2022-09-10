import numpy as np

import cvxopt
from cvxopt import matrix
from cvxopt import solvers

def solvers_qp(vcov,mu,optimal_portfolio,inequality=False,):
    """
    Definimos la función solvers_qp que utiliza el solvers.qp del módulo cvxopt

    Optimiza un problema de la forma 

    min (1/2)x'Px + q'x
    s.a Gx <= h
        Ax = b

    Si adaptamos esta optimización a nuestro ejercicio de optimización del
    portafolio óptimo entonces tenemos que 
    
    Si inequality = False

        min L = (1/2)w'Pw
        s.a w'E = E(rp)
            w'1 = 1
    Si inequality = True

        min L = (1/2)w'Pw
        s.a w'E = E(rp)
            w'1 >= 0
    
    Es decir, podemos restringir la optimización para que la suma de las ponderaciones
    sea uno o bien, para que sea no negativa 

    Parámetros:
    -------------------------------------------------------------------

    vcov : Matriz (n x n) de varianzas y covarianzas


    mu : vector de tamaño (n,) de los retornos esperados de los n activos

    optimal_portfolio : Valor del retorno óptimo, este debe estar dentro de la frontera eficiente
                        puede ser el retorno del portafolio de mínima varianza global, el portafolio
                        óptimo de sharpe, o cualquier otro. 

    inequality : bool , default = False si w'1 = 1 o True si w'1 >= 0

    Ejemplo: 
    -------------------------------------------------------------------
    vcov : (5x5)

        [[0.02450049 0.00388752 0.00228452 0.00095474 0.00308689]
        [0.00388752 0.00874613 0.00182295 0.00094029 0.00445079]
        [0.00228452 0.00182295 0.00385652 0.00142886 0.00119709]
        [0.00095474 0.00094029 0.00142886 0.0021134  0.00087716]
        [0.00308689 0.00445079 0.00119709 0.00087716 0.00734027]]
    mu : (5,)
    
        [0.04029284 0.01008825 0.02163079 0.00754402 0.00529053]

    optimal_portfolio : rpmv

        [0.01000976]

    inequality : False

    [in] solv = solvers_qp(vcov = vcov, mu = mu, optimal_portfolio=rpt, inequality=True)
        print(solv)

    [out]
        [ 1.26e-02]
        [ 3.92e-02]
        [ 1.56e-01]
        [ 6.81e-01]
        [ 1.10e-01] 
    """
    n = len(mu)
    if inequality == False:
        P = matrix(vcov)
        q = matrix(np.zeros((n,1)))
        G = matrix(np.concatenate((
            -np.transpose(np.array(mu)).reshape((n,1)),
            -np.ones(n).reshape(n,1)),1).T)
        h = matrix(-np.array([optimal_portfolio,[1]]))
    elif inequality == True:
        P = matrix(vcov)
        q = matrix(np.zeros((n,1)))
        G = matrix(np.concatenate((
            -np.transpose(np.array(mu)).reshape((n,1)),
            -np.ones(n).reshape(n,1),
            -np.diag(np.full(n,1))),1).T)
        h = matrix(-np.concatenate((
            np.array([optimal_portfolio,[1]]),
            np.zeros(n).reshape(n,1)),0))

    response = np.array(solvers.qp(P=P,q=q,G=G,h=h,show_progress=False)['x']).reshape(-1)
    return response


def ownSolverQP(returns,mu_0):
    n = returns.shape[1]
    mu = np.reshape(returns.mean(axis=0)*12,(1,n))
    cov = np.cov(returns.T,ddof=1)*12

    ones = np.ones((1,n))
    zeros = np.zeros((1,n))

    mu_with_restrictions = np.concatenate((mu,np.array([0,0]).reshape((1,2))),axis=1)
    ones_with_restrictions = np.concatenate((ones,np.array([0,0]).reshape((1,2))),axis=1)
    zeros_with_restrictions = np.concatenate((zeros,np.array([mu_0,1]).reshape((1,2))),axis=1)

    C = np.concatenate((
        cov,
        mu.T,
        ones.T
    ),axis=1)

    C = np.concatenate((
        C,
        mu_with_restrictions,
        ones_with_restrictions
    ),axis=0)

    b = zeros_with_restrictions.T

    x = np.linalg.inv(C) @ b
    return x[:-2,:]