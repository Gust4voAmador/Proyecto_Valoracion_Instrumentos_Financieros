# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:06:25 2024

@author: AMADOR
"""

import yfinance as yf
import pandas as pd
import os
import numpy as np

# %% Métodos
def cargar_datos(etfs, start_date):
    """
    Descarga los datos históricos de una lista de ETFs desde Yahoo Finance.
    """
    return yf.download(etfs, start=start_date)

def calcular_rendimientos_diarios(etf_data):
    """
    Calcula los rendimientos logarítmicos diarios de los ETFs.
    """
    r_diario = np.log(etf_data['Close'] / etf_data['Close'].shift(1))
    r_diario = pd.DataFrame(r_diario).dropna()  # Eliminar filas con valores NaN
    return r_diario

def agregar_informacion(r_diario):
    """
    Agrega columnas con la suma del portafolio y el año a los rendimientos diarios.
    """
    r_diario['portafolio'] = r_diario.sum(axis=1)
    r_diario = r_diario.reset_index()  # Asegurar que 'Date' no es índice
    r_diario['Date'] = pd.to_datetime(r_diario['Date'])  # Convertir a datetime
    r_diario['Year'] = r_diario['Date'].dt.year  # Agregar columna con el año
    return r_diario

def calcular_media_anual(r_diario):
    """
    Calcula la media anual de los rendimientos diarios.
    """
    media_r_diario = r_diario.groupby('Year').mean().drop(columns=['Date'])
    return media_r_diario

def contar_observaciones_anuales(r_diario):
    """
    Cuenta el número de observaciones por año (D)
    """
    obs_por_anno = r_diario.groupby('Year').size().reset_index(name='obs')
    return obs_por_anno.set_index('Year')

def calcular_covarianza(x, y, year_column, year_obs):
    """
    Calcula la covarianza diaria y anual entre dos activos.
    """
    data = pd.DataFrame({'x': x, 'y': y, 'Year': year_column})
    media_x = data.groupby('Year')['x'].mean()
    media_y = data.groupby('Year')['y'].mean()

    def cov_diaria(df):
        return ((df['x'] - media_x[df.name]) * (df['y'] - media_y[df.name])).mean()

    cov_diaria_por_anio = data.groupby('Year').apply(cov_diaria)
    cov_anual_por_anio = cov_diaria_por_anio * year_obs['obs']

    return pd.DataFrame({
        'Covarianza_diaria': cov_diaria_por_anio,
        'Covarianza_anual': cov_anual_por_anio
    })

# %% Flujo principal
def main():
    # Configuración
    etfs = ['XLK', 'XLV', 'XLF', 'EFA', 'VNQ', 'ICLN', 'JNK', 'LQD', 'DBC', 'VWO']
    start_date = '2010-01-01'

    # Carga de datos
    etf_data = cargar_datos(etfs, start_date)

    # Cálculo de rendimientos diarios
    r_diario = calcular_rendimientos_diarios(etf_data)

    # Agregar información
    r_diario = agregar_informacion(r_diario)

    # Media y observaciones anuales
    media_r_diario = calcular_media_anual(r_diario)
    obs_por_anno = contar_observaciones_anuales(r_diario)

    # Cálculo de covarianza
    rendimientos_activo1 = r_diario['XLK']
    rendimientos_activo2 = r_diario['XLV']
    cov_result = calcular_covarianza(
        rendimientos_activo1, 
        rendimientos_activo2, 
        r_diario['Year'], 
        obs_por_anno
    )

    # Resultados
    print("Media anual de rendimientos:")
    print(media_r_diario)
    print("\nCovarianza diaria y anual:")
    print(cov_result)

if __name__ == "__main__":
    main()
