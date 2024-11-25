# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:06:25 2024

@author: AMADOR
"""

import yfinance as yf
import pandas as pd
import numpy as np
import math
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
    return r_diario

def media_diaria_retornos(r_diario):
    """
    A partir de un df de rendimientos calcula la media de los activos, manteniendo los activos como columnas.
    """
    media_diaria = r_diario.mean()
    media_diaria_df = pd.DataFrame(media_diaria).T  # Convertir la serie resultante a DataFrame y transponer

    #print(media_diaria_df)
    return media_diaria_df


def calcular_media_anual(r_diario):
    """
    Calcula la media anual de los rendimientos diarios.
    """
    media_anual = media_diaria_retornos(r_diario) * 252
    return media_anual

def desviacion_std_anual(r_diario):
    """
    Estimación anual de la desviación estándar de retornos.
    """
    var_diaria = r_diario.var()
    var_anual = var_diaria * 252
    std_anual = np.sqrt(var_anual)
    return std_anual

def covarianza_anual(r_diario):
    """
    Calcula la covarianza anual entre los activos.
    """
    cov_diario = r_diario.cov()
    cov_anual = cov_diario * 252
    return cov_anual

def cov_anual_libre_prestamo(r_diario):
    """
    Calcula la covarianza considerando cero la libre de riesgo y pedir prestado
    """
    cov_anual = covarianza_anual(r_diario)
    
    # Crear un DataFrame con dos filas de ceros y el mismo número de columnas que df
    filas_ceros = pd.DataFrame(np.zeros((2, cov_anual.shape[1])), columns=cov_anual.columns)
    
    # Concatenar el DataFrame de ceros al inicio del DataFrame original
    cov_ajuste = pd.concat([filas_ceros, cov_anual], ignore_index=True)
    
    # Agregar columnas al inicio
    cov_ajuste.insert(0, 'libre_riesgo', 0)  
    cov_ajuste.insert(1, 'prestamo', 0)  
    
    return cov_ajuste

def media_anual_ajuste(r_diario, tasa_libre, tasa_prestamo):
    """
    Calcula la media con la tasa libre de riesgo y la tasa de préstamo
    """
    media_anual_a = calcular_media_anual(r_diario)
    # Insertar espacios con las tasas
    media_anual_a.insert(0, 'libre_riesgo', tasa_libre)  
    media_anual_a.insert(1, 'prestamo', tasa_prestamo)  
    return media_anual_a

#%%
#Probar las funciones

# Configuración
#etfs = ['XLK', 'XLV', 'XLF', 'EFA', 'VNQ', 'ICLN', 'JNK', 'LQD', 'DBC', 'VWO']
#start_date = '2010-01-01'

# 1. Carga de datos
#print("Descargando datos históricos de ETFs...")
#etf_data = cargar_datos(etfs, start_date)

# 2. Cálculo de rendimientos diarios
#print("Calculando rendimientos diarios...")
#r_diario = calcular_rendimientos_diarios(etf_data)

#cov_ajutada = cov_anual_libre_prestamo(r_diario)
#media_anual = calcular_media_anual(r_diario)
#media_diaria = media_diaria_retornos(r_diario)
#media_anual_param = media_anual_ajuste(r_diario, 0.12, 0.15)


#%%
def minimizar_varianza(media_objetivo, media_anual, cov_anual):
    """
    Minimiza la varianza del portafolio para un retorno deseado.
    """
    # Verificar tamaños
    media_anual = np.squeeze(np.asarray(media_anual))
    
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
    # Verificar tamaños (se deja en forma vector y no matriz)
    media_anual = np.squeeze(np.asarray(media_anual))
    
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
    """
    Minimiza la varianza con venta en corto.
    """
    # Verificar tamaños
    media_anual = np.squeeze(np.asarray(media_anual))
    
    # Función a minimizar
    def objetivo(pesos):
        return np.dot(pesos.T, np.dot(cov_anual, pesos))
    
    # Verificar llegar al retorno esperado
    def rest_esperado(pesos):
        return np.dot(pesos, media_anual) - esperado
    
    # Verificar que sume 1
    def rest_suma(pesos):
        return np.sum(pesos) - 1

    restricciones = [
        {'type': 'eq', 'fun': rest_esperado},  
        {'type': 'eq', 'fun': rest_suma}       
    ]  

    # Agregar restricciones segun activos especificos
    if activos:
        for indice, (cota_inf, cota_sup) in activos.items():
            if cota_inf == cota_sup and cota_sup is not None:
                # Poner el valor estricto
                restricciones.append({'type': 'eq', 'fun': lambda pesos, i=indice: pesos[i] - cota_inf})
            else:
                # Para un rango
                restricciones.append({'type': 'ineq', 'fun': lambda pesos, i=indice: pesos[i] - cota_inf})
                if cota_sup is not None:
                    restricciones.append({'type': 'ineq', 'fun': lambda pesos, i=indice: cota_sup - pesos[i]})

    # Si se vende en corto se permite el endeudamiento
    if corto:
        intervalo = [(-1, 1) for _ in range(len(media_anual))]  
    else:
        intervalo = [(0, 1) for _ in range(len(media_anual))]   

    # Pesos iniciales
    iniciales = np.array([1 / len(media_anual)] * len(media_anual))

    # Resultados
    resultado = minimize(
        objetivo,
        iniciales,
        method='SLSQP',
        bounds=intervalo,
        constraints=restricciones
    )

    return resultado.x
# %%MonteCarlo

def simulacion_portafolio_montecarlo(pesos, retornos, S0, covarianza, num_simulaciones=1000, num_periodos=252):
    """
    Simula el rendimiento de un portafolio usando Monte Carlo y  a partir del movimiento browniano geométrico.
    
    Return:
        retornos_simulados: np.ndarray
            Array con los retornos simulados del portafolio al final del año.
    """

    # Ajuste de parámetros para la escala temporal (diaria en este caso)
    retornos_diarios = retornos / num_periodos
    covarianza_diaria = covarianza / num_periodos
    
    #Valor inicial del portafolio
    valor_inicial_portafolio = np.dot(pesos, S0)
    
    # Descomposición de Cholesky para generar variables correlacionadas
    L = np.linalg.cholesky(covarianza_diaria)

    num_activos = len(pesos)
    valores_portafolio = np.zeros(num_simulaciones)
    retornos_simulados = np.zeros(num_simulaciones)

    for sim in range(num_simulaciones):
        precios = S0  # Precios iniciales
        for t in range(num_periodos):
            # Generar ruido aleatorio correlacionado
            z = np.random.normal(0, 1, num_activos)
            variacion = retornos_diarios - 0.5 * np.diag(covarianza_diaria)
            ruido_correlacionado = z @ L.T
            # Actualizar precios
            precios *= np.exp(variacion + ruido_correlacionado)

        # Valor del portafolio al final
        valores_portafolio[sim] = np.dot(pesos, precios)
        retornos_simulados = math.log(valores_portafolio[sim]/valor_inicial_portafolio)
        

    return retornos_simulados


# %% Métricas

def sharpe_ratio(retorno_portafolio, tasa_libre, desviacion_portafolio):
    """
    Calcula el Sharpe Ratio.
    """
    return (retorno_portafolio - tasa_libre) / desviacion_portafolio


def safety_first_ratio(retorno_portafolio, retorno_umbral, desviacion_portafolio):
    """
    Calcula el Roy's Safety First Ratio.
    """
    return (retorno_portafolio - retorno_umbral) / desviacion_portafolio


def sortino_ratio(retorno_portafolio, tasa_libre, desviacion_downside):
    """
    Calcula el Sortino Ratio.
    """
    return (retorno_portafolio - tasa_libre) / desviacion_downside


def treynor_ratio(retorno_portafolio, tasa_libre, beta_portafolio):
    """
    Calcula el Treynor Ratio.

    """
    return (retorno_portafolio - tasa_libre) / beta_portafolio


def jensens_alpha(retorno_portafolio, etfs, start_date, tasa_libre, pesos):
    """
    Calcula el Jensen's Alpha.

    Parámetros:
    - retorno_portafolio: Retorno promedio del portafolio.
    - tasa_libre: Retorno promedio libre de riesgo.
    - beta_portafolio: Beta del portafolio.
    - retorno_mercado: Retorno promedio del mercado.

    Retorna:
    - Jensen's Alpha.
    """
    CAPM = capm(etfs, start_date, tasa_libre, pesos)
    retorno_esperado = CAPM['rend_port']
    return retorno_portafolio - retorno_esperado


def capm(etfs, start_date, tasa_libre, pesos):
    """
    Calcula el Capm del portafolio
    """
    
    # Cargar los datos del mercado 
    mercado = cargar_datos("^GSPC", start_date)
    
    # Cargar los Etfs
    etf_data = cargar_datos(etfs, start_date)
    
    # Rendimientos 
    r_mercado = calcular_rendimientos_diarios(mercado)
    r_diario = calcular_rendimientos_diarios(etf_data)
    
    # Calcular los betas individuales con la pendiente de la regresión
    betas = [0,0]
    for activo in r_diario.columns:
        covarianza = np.cov(r_diario[activo], r_mercado["^GSPC"])[0, 1]
        var_mercado = np.var(r_mercado["^GSPC"])
        beta = covarianza / var_mercado
        betas.append(beta)
        
    # Ver los betas
    activos = ["tasa libre", "tasa prestamo"] + list(r_diario.columns)
    betas_df = pd.DataFrame({'Activo': activos, 'Beta': betas})
    
    # Beta del portafolio
    beta_port = np.dot(pesos, betas)

    # Prima de riesgo
    prima_riesgo = r_mercado["^GSPC"].mean()*252 - tasa_libre

    # Retorno esperado
    rend_port = tasa_libre + beta_port * prima_riesgo
    
    return {
        'betas': betas_df,
        'beta_port': beta_port,
        'rend_port': rend_port
    }
    
# %% Flujo principal

# Configuración
etfs = ['XLK', 'XLV', 'XLF', 'EFA', 'VNQ', 'ICLN', 'JNK', 'LQD', 'DBC', 'VWO']
start_date = '2010-01-01'

# Carga de datos
etf_data = cargar_datos(etfs, start_date)

# Cálculo de rendimientos diarios
r_diario = calcular_rendimientos_diarios(etf_data)

# Cálculo de media y desviación estándar anual
media_anual = media_anual_ajuste(r_diario, 0.043, 0.0531)
desviacion_anual = desviacion_std_anual(r_diario)

# Cálculo de covarianza anual
cov_anual = cov_anual_libre_prestamo(r_diario)

# Portafolio Conservador 
esp_cons = 0.05
activos_cons = {0: (0, None), # La tasa libre sin cota
           1: (0, 0), # No permitir apalancamiento
           2: (0, 0.2), # Limitar el resto de activos a no más de un 20%
           3: (0, 0.2),
           4: (0, 0.2),
           5: (0, 0.2),
           6: (0, 0.2),
           7: (0, 0.2),
           8: (0, 0.2),
           9: (0, 0.2),
           10: (0, 0.2),
           11: (0, 0.2)} 
pesos_cons = minimizar2(esp_cons, media_anual, cov_anual, False, activos_cons)
print("\nPesos que optimizan un portafolio conservador:")
print(pesos_cons)

# Portafolio Moderado 
esp_mod = 0.07
activos_mod = {0: (0, None), # La tasa libre sin cota
           1: (0, 0), # No permitir apalancamiento
           2: (0, 0.3), # Limitar el resto de activos a no más de un 30%
           3: (0, 0.3),
           4: (0, 0.3),
           5: (0, 0.3),
           6: (0, 0.3),
           7: (0, 0.3),
           8: (0, 0.3),
           9: (0, 0.3),
           10: (0, 0.3),
           11: (0, 0.3)} 
pesos_mod = minimizar2(esp_mod, media_anual, cov_anual, False, activos_mod)
print("\nPesos que optimizan un portafolio moderado:")
print(pesos_mod)

# Portafolio Agresivo
esp_agr = 0.12
activos_agr = {0: (0, None), # La tasa libre sin cota
           1: (-0.8, 0)} # Permitir el apalancamiento
pesos_agr = minimizar2(esp_agr, media_anual, cov_anual, True, activos_agr)
print("\nPesos que optimizan un portafolio agresivo:")
print(pesos_agr)

# Calculo de Capm y betas
#result_capm = capm(etfs, start_date, 0.05, pesos_min_2)
#print("\nResultados del Capm")
#print(result_capm)


