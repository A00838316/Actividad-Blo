# Actividad-Blo
# Importar librerías necesarias
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Configurar fechas
from datetime import datetime, timedelta
end_date = datetime.now()
start_date_3d = end_date - timedelta(days=3)
start_date_10d = end_date - timedelta(days=10)

# Definir las acciones
tickers = ['AAPL', 'MSFT', 'GOOGL']

# Función para obtener datos corregida
def get_stock_data(tickers, start, end, interval):
    # Obtener datos sin agrupar por ticker primero
    data = yf.download(tickers, start=start, end=end, interval=interval)
    
    # Si hay múltiples tickers, los datos vienen en columnas multi-nivel
    if len(tickers) > 1:
        try:
            # Intentar acceder a 'Adj Close' desde columnas multi-nivel
            df = data['Adj Close']
        except KeyError:
            # Si falla, usar 'Close' como respaldo
            df = data['Close']
    else:
        # Para un solo ticker
        df = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    # Asegurarse de que sea un DataFrame con los nombres de tickers correctos
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df, columns=[tickers[0]])
    elif len(tickers) > 1 and set(df.columns) != set(tickers):
        df.columns = tickers
    
    return df

# Obtener datos
data_5min_3days = get_stock_data(tickers, start_date_3d, end_date, '5m')
data_30min_10days = get_stock_data(tickers, start_date_10d, end_date, '30m')

# Función para prueba de raíz unitaria (ADF Test)
def adf_test(series, title=''):
    print(f'\nPrueba ADF para {title}:')
    result = adfuller(series.dropna())
    print(f'Estadístico ADF: {result[0]}')
    print(f'p-valor: {result[1]}')
    print('Valores críticos:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    if result[1] <= 0.05:
        print("Conclusión: Serie estacionaria (rechaza H0)")
    else:
        print("Conclusión: Serie no estacionaria (no rechaza H0)")

# Función para prueba de cointegración
def cointegration_test(df):
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            print(f'\nPrueba de cointegración entre {tickers[i]} y {tickers[j]}:')
            score, p_value, _ = coint(df[tickers[i]], df[tickers[j]])
            print(f'Estadístico: {score}')
            print(f'p-valor: {p_value}')
            if p_value < 0.05:
                print("Conclusión: Hay cointegración")
            else:
                print("Conclusión: No hay cointegración")

# Análisis para 5min 3 días
print("=== Análisis 5 minutos - 3 días ===")
for ticker in tickers:
    adf_test(data_5min_3days[ticker], f'{ticker} (5min-3d)')

cointegration_test(data_5min_3days)

for ticker in tickers:
    print(f'\nModelo ARMA para {ticker} (5min-3d):')
    series = data_5min_3days[ticker].dropna()
    model = ARIMA(series, order=(1,0,1))
    results = model.fit()
    print(results.summary())

# Análisis para 30min 10 días
print("\n=== Análisis 30 minutos - 10 días ===")
for ticker in tickers:
    adf_test(data_30min_10days[ticker], f'{ticker} (30min-10d)')

cointegration_test(data_30min_10days)

for ticker in tickers:
    print(f'\nModelo ARMA para {ticker} (30min-10d):')
    series = data_30min_10days[ticker].dropna()
    model = ARIMA(series, order=(1,0,1))
    results = model.fit()
    print(results.summary())

# Visualización
plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
data_5min_3days.plot(title='Precios 5min-3días')
plt.subplot(2,1,2)
data_30min_10days.plot(title='Precios 30min-10días')
plt.tight_layout()
plt.show()


import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from google.colab import files

end_date = datetime.now()
start_date_3d = end_date - timedelta(days=3)
start_date_10d = end_date - timedelta(days=10)

tickers = ['AAPL', 'MSFT', 'GOOGL']

def get_stock_data(tickers, start, end, interval):
    data = yf.download(tickers, start=start, end=end, interval=interval)
    if len(tickers) > 1:
        try:
            df = data['Adj Close']
        except KeyError:
            df = data['Close']
    else:
        df = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df, columns=[tickers[0]])
    elif len(tickers) > 1 and set(df.columns) != set(tickers):
        df.columns = tickers
    
    # Remover la información de zona horaria del índice
    df.index = df.index.tz_localize(None)
    
    return df

data_5min_3days = get_stock_data(tickers, start_date_3d, end_date, '5m')
data_30min_10days = get_stock_data(tickers, start_date_10d, end_date, '30m')

excel_5min_3days = 'stock_data_5min_3days.xlsx'
data_5min_3days.to_excel(excel_5min_3days)
files.download(excel_5min_3days)
print(f"Archivo {excel_5min_3days} creado y descargado")

excel_30min_10days = 'stock_data_30min_10days.xlsx'
data_30min_10days.to_excel(excel_30min_10days)
files.download(excel_30min_10days)
print(f"Archivo {excel_30min_10days} creado y descargado")

excel_combined = 'stock_data_combined.xlsx'
with pd.ExcelWriter(excel_combined) as writer:
    data_5min_3days.to_excel(writer, sheet_name='5min_3days')
    data_30min_10days.to_excel(writer, sheet_name='30min_10days')
files.download(excel_combined)
print(f"Archivo combinado {excel_combined} creado y descargado")
