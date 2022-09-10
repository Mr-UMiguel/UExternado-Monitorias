import pandas as pd
import numpy as np
import yfinance as yf


def get_prices(symbols,start,end, intervals):
    """
    Función que retorna un array con el precio ajustado para cada símbolo

    Parámetros:
        symbols: list=['AAPL','TSLA']
        start: str='YYYY-MM-DD'
        end: str='YYYY-MM-DD'
        intervals: str
            valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    Ejemplo: 

    prices = get_prices(['AAPL','TSLA'],start='2021-01-01',end='2022-01-01',intervals='M')

    """
    data = yf.download(symbols,start=start,end=end,progress=False, intervals=intervals)
    prices = data['Adj Close']
    
    return prices