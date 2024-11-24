# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 01:24:00 2024

@author: AMADOR
"""

import yfinance as yf
import pandas as pd
import os
import numpy as np

# %% 
# Carga de datos

# Definir los símbolos de los ETFs
etfs = ['XLK', 'XLV', 'XLF', 'EFA', 'VNQ', 'ICLN', 'JNK', 'LQD', 'DBC', 'VWO']

# Descargar los datos históricos de los ETFs
etf_data10 = yf.download(etfs, start='2010-01-01')

# Obtener la ruta relativa para la carpeta 'data' en el directorio padre
ruta_carpeta_data = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
#os.makedirs(ruta_carpeta_data, exist_ok=True)  

ruta_csv = os.path.join(ruta_carpeta_data, 'etf_data10.csv')

# Guardar los datos en un archivo CSV
#etf_data10.to_csv(ruta_csv)

# %%
# Estimaci´on de la media anual de los retornos



# Rendimiento diario en el día t
r_diario = np.log(etf_data10['Close'] / etf_data10['Close'].shift(1))
r_diario = pd.DataFrame(r_diario) #Hacerlo df
print(r_diario.head())

# Eliminar primera observación
r_diario = r_diario.tail(-1)

# Agregar columna portafolio (redimiento total de todos los activos)
r_diario['portafolio'] = r_diario.sum(axis=1)


# Rendimiento promedio diario

# Reconocer la columna Date
r_diario = r_diario.reset_index()
print(r_diario.columns)

# Pasar a formato datetime
r_diario['Date'] = pd.to_datetime(r_diario['Date']) 

# Crear una columna con años
r_diario['Year'] = r_diario['Date'].dt.year

# Agrupar por año y calcular la media anual
media_r_diario = r_diario.groupby('Year').mean()

#Eliminar columna Date
media_r_diario= media_r_diario.drop(columns = ['Date'])



# Agrupar por 'Year' y contar observaciones
obs_por_anno = r_diario.groupby('Year').size().reset_index(name='obs')

# Establecer 'Year' como índice
obs_por_anno = obs_por_anno.set_index('Year')

# Media de rendimiento anual
media_r_anual = media_r_diario.mul(obs_por_anno['obs'], axis=0)

# %%
# C´alculo de la desviaci´on est´andar anual

# Crear df con la columna año
var_diario = pd.DataFrame({'Year': r_diario['Year']})

# Varianza diaria 1 a 10
var_diario['var_portafolio'] = r_diario.iloc[:, 1:11].var(axis=1)

# Varianza anual
var_anual = var_diario.groupby('Year').sum()

# Desviación anual
desv_anual = pd.DataFrame({'desv_portafolio': var_anual['var_portafolio']})
desv_anual = np.sqrt(desv_anual)

#%% 
#Calcular covarianza
def calcular_covarianza(x, y, year_column, year_obs):
    """
    Calcula la covarianza diaria y anual entre dos activos.

    Parámetros:
    - x: Serie o columna de pandas con los rendimientos diarios del primer activo.
    - y: Serie o columna de pandas con los rendimientos diarios del segundo activo.
    - year_column: Serie o columna que indica el año de cada observación.
    - year_obs: DataFrame con la cantidad de observaciones por año (columna `obs` con índice `Year`).

    Retorna:
    - DataFrame con la covarianza diaria y anual por cada año.
    """
    # Crear un DataFrame con los valores
    data = pd.DataFrame({'x': x, 'y': y, 'Year': year_column})

    # Calcular la media diaria por año
    media_x = data.groupby('Year')['x'].mean()
    media_y = data.groupby('Year')['y'].mean()

    # Calcular la covarianza diaria por año
    def cov_diaria(df):
        return ((df['x'] - media_x[df.name]) * (df['y'] - media_y[df.name])).mean()

    cov_diaria_por_anio = data.groupby('Year').apply(cov_diaria)

    # Calcular la covarianza anual usando el número de observaciones por año
    cov_anual_por_anio = cov_diaria_por_anio * year_obs['obs']

    # Crear el DataFrame final
    result = pd.DataFrame({
        'Covarianza_diaria': cov_diaria_por_anio,
        'Covarianza_anual': cov_anual_por_anio
    })

    return result



# eejemplo
rendimientos_activo1 = r_diario['XLK']  # Columna del activo 1
rendimientos_activo2 = r_diario['XLV']  # Columna del activo 2

# Cálculos
cov_result = calcular_covarianza(
    rendimientos_activo1, 
    rendimientos_activo2, 
    r_diario['Year'], 
    obs_por_anno
)

print(cov_result)








