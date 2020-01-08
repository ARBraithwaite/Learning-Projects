import pandas as pd
import numpy as np
import math
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

def plot_ts(series, ts = True, dist = True, pacf=True, acf=True, lags=None):
    """
        Plots the time series, distribution, acf and pacf

        series - time series data
        pass False to not include a plot
        lags - acf, pacf lags
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    fig, axis = plt.subplots(figsize=(15,10))
    layout= (3, 2)
    
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    dist_ax = plt.subplot2grid(layout, (1, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (2, 0))
    pacf_ax = plt.subplot2grid(layout, (2, 1))
    
    if ts:
        series.plot(ax=ts_ax, title = f'Time Series: {series.name}', legend = False) # TS Plot
    else:
        ts_ax.remove()
    if dist:
        sns.distplot(series, ax = dist_ax)
        dist_ax.set_title('Distribution Plot')
    else:
        dist_ax.remove()
    if acf:
        plot_acf(series, lags = lags, ax=acf_ax)
    else:
        acf_ax.remove()
    if pacf:
        plot_pacf(series, lags = lags, ax=pacf_ax)
    else:
        pacf_ax.remove()
    
    plt.tight_layout()

def box_plots(series, y: str):
    """
        Plot time series box plot visualisations

        Series - ts data
        y - name of the target variable
    """
    fig, ax = plt.subplots(2, 1, figsize=(15,10))
    sns.boxplot(x=series.index.year, y=y, data=series, ax=ax[0]).set_title("Year-Wise Box Plots")
    sns.boxplot(x=series.index.month, y=y, data=series, ax=ax[1]).set_title("Month-Wise Box Plots")
    for ax in ax:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()

def reg_trend(X, y, degree: int):
    """
        linear, polynomial trends or higher order trends

        Arrays X and y shape (len, 1), degree for regression fit
        degree - order of fit
    """
    
    #Imports
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    #Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    lr_trend = lr_model.predict(X)

    # Polynomial Regression
    ply_model = Pipeline([('polynomial_features', PolynomialFeatures(degree=degree)),
                      ('lr' , LinearRegression())
                     ])
    ply_model.fit(X, y)
    ply_trend = ply_model.predict(X)

    return lr_trend, ply_trend

def decompose(series, model='additive'):
    """
        Decompose ts using statsmodels method

        series - ts data
        model - type of ts 'multiplicative' or 'additive'
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    result = seasonal_decompose(series, model=model)
    
    fig, axis = plt.subplots(4, 1, figsize=(15,10))
    
    result.observed.plot(ax=axis[0], title='Observed')
    result.trend.plot(ax=axis[1], title='Trend')
    result.seasonal.plot(ax=axis[2], title='Seasonal')
    result.resid.plot(ax=axis[3], title='Residual')
    
    plt.tight_layout()

def stl_decompose(series, period=None):
    """
        Decompose ts using STL - Seasonal and Trend decomposition using Loess

        series - ts data
        period - (largest) period of seasonality
    """
    import stldecompose as stl
    
    result = stl.decompose(series, period=period)
    
    fig, axis = plt.subplots(4, 1, figsize=(15,10))
    
    result.observed.plot(ax=axis[0], title='Observed')
    result.trend.plot(ax=axis[1], title='Trend')
    result.seasonal.plot(ax=axis[2], title='Seasonal')
    result.resid.plot(ax=axis[3], title='Residual')
    
    plt.tight_layout()

def test_stationarity(series):
    """
        ADF test
        
        series - time series data
    """
    
    #Perform Dickey-Fuller test:
    print('\n'+series.name)
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
def adfuller_test(series, signif=0.05, name='', verbose=False):
    """
        ADF test
        
        series - time series data
        name - column name
        signif = significance value
    """
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print("\n")
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series may be Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is likely Non-Stationary.")  

def optimize_SARIMA_ARIMA(series, parameters_list, d, D, s, mode='SARIMA'):
    """
        Return dataframe with parameters and corresponding AIC
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order 
        s - length of season
    """
    results = []
    best_aic = float("inf")

    for param in parameters_list:
        # we need try-except because on some combinations model fails to converge
        try:
            if mode == 'SARIMA':
                model = SARIMAX(series, order=(param[0], d, param[1]), 
                                            seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
            elif mode == 'ARIMA':
                model = SARIMAX(series, order=(param[0],d,param[1])).fit(disp=-1)
        except:
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def precision_measure(y, y_pred):
    return sum((y_pred-y)**2)/sum((y-np.mean(y_pred))**2)

def plot_SARIMA_ARIMA_forecast(series, model, n_steps, s=0, d=0):
    """
        Plots model vs predicted values
        
        series - dataset with timeseries
        model - fitted SARIMA/ARIMA model
        n_steps - number of steps to predict in the future
        s - seaonal window
        
    """
    # adding model values
    data = pd.DataFrame()
    data['actual'] = series.copy()
    data['arima_model'] = model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differencing
    data['arima_model'][:s+d] = np.NaN
    
    # forecasting on n_steps forward 
    forecast = model.get_prediction(start = data.shape[0], end = data.shape[0]+n_steps, full_results=True)
    lower_bound = forecast.conf_int().iloc[:,0]
    upper_bound = forecast.conf_int().iloc[:,1]
    forecast = data.arima_model.append(forecast.predicted_mean)
    
    # calculate error, again having shifted on s+d steps from the beginning
    mape = mean_absolute_percentage_error(data['actual'][s+d:], data['arima_model'][s+d:])
    rmse = math.sqrt(mean_squared_error(data['actual'][s+d:], data['arima_model'][s+d:]))
    mae = mean_absolute_error(data['actual'][s+d:], data['arima_model'][s+d:])
    pm = precision_measure(data['actual'][s+d:], data['arima_model'][s+d:])

    plt.figure(figsize=(20, 10))
    plt.title(f'MAPE: {mape:.2f} - RMSE: {rmse:.2f} - MAE: {mae:.2f} - PM: {pm:.2f}')
    plt.plot(forecast, color='r', label="model")
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.fill_between(lower_bound.index, 
                 lower_bound, 
                 upper_bound, 
                 color='darkgrey', alpha=.5)
    plt.plot(data.actual, label="actual")
    plt.legend()
    plt.grid(True)
    
def plot_VAR_forecast(series, model, n_steps, d=0):
    """
        Plots model vs predicted values
        
        series - dataset with timeseries
        model - fitted VAR model
        n_steps - number of steps to predict in the future
        d - difference
        
    """
    # adding model values
    data = pd.DataFrame()
    data['actual'] = series.copy()
    # Inverse difference transform - assuming first order
    data['var_model'] = series.iloc[0] + model.fittedvalues[series.name].cumsum()
    
    # forecasting on n_steps forward 
    forecast = model.get_forecast(n_steps)
    forecast = series.iloc[-1] + forecast.predicted_mean.cumsum()
    forecast = data.var_model.append(forecast[series.name])
    
    # calculate error, again having shifted on s+d steps from the beginning
    mape = mean_absolute_percentage_error(data['actual'][d:], data['var_model'][d:])
    rmse = math.sqrt(mean_squared_error(data['actual'][d:], data['var_model'][d:]))
    mae = mean_absolute_error(data['actual'][d:], data['var_model'][d:])
    pm = precision_measure(data['actual'][d:], data['var_model'][d:])

    plt.figure(figsize=(20, 10))
    plt.title(f'MAPE: {mape:.2f} - RMSE: {rmse:.2f} - MAE: {mae:.2f} - PM: {pm:.2f}')
    plt.plot(forecast, color='r', label="model")
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data.actual, label="actual")
    plt.legend()
    plt.grid(True)