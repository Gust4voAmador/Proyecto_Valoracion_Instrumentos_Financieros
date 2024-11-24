# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:06:25 2024

@author: AMADOR
"""

import yfinance as yf
import pandas as pd
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
    r_diario['Portafolio'] = r_diario.sum(axis=1)
    return r_diario

def media_diaria_retornos(r_diario):
    """
    A partir de un df de rendimientos calcula media de activos y el portafolio
    """
    media_diaria = r_diario.mean()
    return media_diaria

def calcular_media_anual(r_diario):
    """
    Calcula la media anual de los rendimientos diarios.
    """
    media_anual = media_diaria_retornos(r_diario) * 252
    return media_anual

def desviacion_std_anual(r_diario):
    """
    Estimación anual de la desviación estándar de retornos 
    """
    var_diaria = r_diario.var()
    var_anual = var_diaria * 252
    std_anual = np.sqrt(var_anual) 
    return std_anual

def covarianza_anual(r_diario):
    """
    Calcula la covarianza anual entre los activos.
    """
    # Eliminar columna portafolio
    r_diario = r_diario.drop(columns=['Portafolio'])
    cov_diario = r_diario.cov()
    cov_anual = cov_diario * 252
    return cov_anual

# %% Flujo principal
def main():
    # Configuración
    etfs = ['XLK', 'XLV', 'XLF', 'EFA', 'VNQ', 'ICLN', 'JNK', 'LQD', 'DBC', 'VWO']
    start_date = '2010-01-01'

    # Carga de datos
    etf_data = cargar_datos(etfs, start_date)

    # Cálculo de rendimientos diarios
    r_diario = calcular_rendimientos_diarios(etf_data)

    # Cálculo de media y desviación estándar anual
    media_anual = calcular_media_anual(r_diario)
    desviacion_anual = desviacion_std_anual(r_diario)

    # Cálculo de covarianza anual
    cov_anual = covarianza_anual(r_diario)

    # Resultados
    print("Media anual de rendimientos:")
    print(media_anual)
    print("\nDesviación estándar anual de rendimientos:")
    print(desviacion_anual)
    print("\nMatriz de covarianza anual de los rendimientos:")
    print(cov_anual)

if __name__ == "__main__":
    main()
