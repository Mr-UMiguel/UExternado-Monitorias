import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme()

def get_tsplot(time_series:np.ndarray,dates:list, labels:list, normalize:bool=True):

    if type(time_series) != np.ndarray:
        time_series = np.array(time_series, dtype=np.float64)

    nrows, ncols = time_series.shape

    dates = np.array(dates, dtype=np.datetime64)
    labels = np.array(labels)

    if len(dates) != nrows:
        raise Exception("la longitud de dates debe ser igual al número de filas de time_series")
    
    if len(labels) != ncols:
        raise Exception("la longitud de labels debe ser igual al número de columnas de time_series")

    if normalize:
        vmin = np.apply_along_axis(np.min, arr=time_series,axis=0)
        vmax = np.apply_along_axis(np.max, arr=time_series,axis=0)

        ## Normalized time series
        time_series = np.array([(x - vmin)/(vmax-vmin) for x in time_series])

    fig = plt.figure(figsize=(10,7))
    for col in range(ncols):
        sns.lineplot(x=dates,y=time_series[:,col])
    plt.legend(labels,loc='upper left')
    plt.show()