# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:06:25 2024

@author: AMADOR
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize

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
    Calcula la covarianza anual entre los activos, excluyendo la columna 'Portafolio'.
    """
    r_diario = r_diario.drop(columns=['Portafolio'])  # Excluir columna 'Portafolio'
    cov_diario = r_diario.cov()
    cov_anual = cov_diario * 252
    return cov_anual


#%%
def minimizar_varianza(media_objetivo, media_anual, cov_anual):
    """
    Minimiza la varianza del portafolio para un retorno deseado.
    """
    # Asegurarse de que media_anual y cov_anual excluyen 'Portafolio'
    media_anual = media_anual.drop('Portafolio')
    n = len(media_anual)
    w0 = np.ones(n) / n  # Pesos iniciales iguales

    # Restricciones: 1) Retorno deseado y 2) Suma de pesos igual a 1
    restricciones = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Suma de pesos = 1
        {'type': 'eq', 'fun': lambda w: np.dot(w, media_anual) - media_objetivo}  # Retorno deseado
    )

    # Función objetivo: minimizar la varianza
    def funcion_objetivo(w):
        return np.dot(w.T, np.dot(cov_anual, w))

    resultado = minimize(funcion_objetivo, w0, method='SLSQP', constraints=restricciones)
    return resultado.x

def maximizar_retorno(varianza_objetivo, media_anual, cov_anual):
    """
    Maximiza el retorno del portafolio para una varianza deseada.
    """
    # Asegurarse de que media_anual y cov_anual excluyen 'Portafolio'
    media_anual = media_anual.drop('Portafolio')
    n = len(media_anual)
    w0 = np.ones(n) / n  # Pesos iniciales iguales

    # Restricciones: 1) Varianza deseada y 2) Suma de pesos igual a 1
    restricciones = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Suma de pesos = 1
        {'type': 'eq', 'fun': lambda w: np.dot(w.T, np.dot(cov_anual, w)) - varianza_objetivo}  # Varianza deseada
    )

    # Función objetivo: maximizar el retorno (equivalente a minimizar -retorno)
    def funcion_objetivo(w):
        return -np.dot(w, media_anual)

    resultado = minimize(funcion_objetivo, w0, method='SLSQP', constraints=restricciones)
    return resultado.x

def minimizar2(esperado, media_anual, cov_anual, corto, activos):
    media_anual = media_anual.drop('Portafolio')
    
    def objetivo(pesos):
        return np.dot(pesos.T, np.dot(cov_anual, pesos))

    def rest_esperado(pesos):
        return np.dot(pesos, media_anual) - esperado
    
    def rest_suma(pesos):
        return np.sum(pesos) - 1

    restricciones = [
        {'type': 'eq', 'fun': rest_esperado},  
        {'type': 'eq', 'fun': rest_suma}       
    ]  # Target return constraint

    
    if activos:
        for indice, (cota_inf, cota_sup) in activos.items():
            restricciones.append({'type': 'ineq', 'fun': lambda pesos, i=indice: pesos[i] - cota_inf})
            if cota_sup is not None:
                restricciones.append({'type': 'ineq', 'fun': lambda pesos, i=indice: cota_inf - pesos[i]})

    # Set bounds for weights
    if corto:
        intervalo = [(-1, 1) for _ in range(len(media_anual))]  
    else:
        intervalo = [(0, 2) for _ in range(len(media_anual))]   

    
    iniciales = np.array([1 / len(media_anual)] * len(media_anual))

    # Perform optimization
    resultado = minimize(
        objetivo,
        iniciales,
        method='SLSQP',
        bounds=intervalo,
        constraints=restricciones
    )

    return resultado.x
    
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

    # Primer método: minimizar la varianza dado un retorno deseado
    media_objetivo = 0.10  # Supongamos un retorno deseado del 10% anual
    pesos_min_var = minimizar_varianza(media_objetivo, media_anual, cov_anual)
    print("\nPesos que minimizan la varianza dado un retorno deseado del 10%:")
    print(pesos_min_var)

    # Segundo método: maximizar el retorno dado una varianza deseada
    varianza_objetivo = 0.05  # Supongamos una varianza deseada de 0.05
    pesos_max_ret = maximizar_retorno(varianza_objetivo, media_anual, cov_anual)
    print("\nPesos que maximizan el retorno dado una varianza deseada de 0.05:")
    print(pesos_max_ret)
    
    # Tercer método: minimizar con venta en corto y apalancamiento
    esperado = 0.10
    activos = {0: (0, None), # El activo 1 positivo
               1: (-1, 0)} # El activo 2 negativo pero controlado
    pesos_min_2 = minimizar2(esperado, media_anual, cov_anual, True, activos)
    print("\nPesos que minimizan la varianza dado un retorno deseado del 10%:")
    print(pesos_min_2)
    
if __name__ == "__main__":
    main()
