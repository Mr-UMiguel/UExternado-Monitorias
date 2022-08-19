import pandas as pd
import numpy as np

def get_returns(prices:np.ndarray, type_of_return:str='log'):
    """
    Retorna los precios logaríticos o artiméticos

    Parámetros:
    --------------------------------
        prices:np.array matriz m x n de precios
        type_of_return: default 'log' or 'ari' tipo de retorno deseado
            use 'log' para reotrno logarítmico o continuo
            use 'ari' para retorno aritmético o discreto
    """
    if type(prices) != np.ndarray:
        prices = np.array(prices, dtype=np.float64)

    if (type_of_return != 'log') and (type_of_return != 'ari'):
        raise ValueError("type_of_return = '{}', no se reconoce como valor válido use 'log' o 'ari' en su lugar".format(type_of_return))

    t0 = np.roll(prices,shift=1,axis=0)[1:]
    t1 = prices[1:]

    if type_of_return == 'log':
        returns = np.log(t1/t0)
    else:
        returns = (t1/t0)-1

    return returns
