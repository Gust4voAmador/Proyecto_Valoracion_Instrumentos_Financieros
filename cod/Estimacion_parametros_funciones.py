# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:06:25 2024

@author: AMADOR
"""
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas as pd
import numpy as np
import math
import random
from scipy.optimize import minimize


#Se establece la semilla
random.seed(123)

# %% Métodos
def cargar_datos(etfs, start_date):
    # Carga de los datos usando yfinance
    data = yf.download(etfs, start=start_date)['Adj Close']
    return data

def calcular_rendimientos_diarios(etf_data):
    """
    Calcula los rendimientos logarítmicos diarios de los ETFs.
    """
    # Calculamos los rendimientos logarítmicos diarios
    r_diario = np.log(etf_data / etf_data.shift(1))
    
    # Eliminamos cualquier fila con valores NaN (por el shift)
    r_diario = r_diario.dropna()  
    
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

def calcular_varianza_portafolio(pesos, cov_anual):
    """
    Calcula la varianza del portafolio dado un vector de pesos y la matriz de covarianza.
    
    Parámetros:
    - pesos: Vector de pesos de los activos en el portafolio.
    - cov_anual: Matriz de covarianza anual de los activos.
    
    Retorna:
    - Varianza del portafolio.
    """
    return np.dot(pesos.T, np.dot(cov_anual, pesos))

def calcular_retorno_esperado_portafolio(pesos, media_anual):
    """
    Calcula el retorno esperado del portafolio dado un vector de pesos y el vector de retornos esperados.
    
    Parámetros:
    - pesos: Vector de pesos de los activos en el portafolio.
    - media_anual: Vector de retornos esperados de los activos.
    
    Retorna:
    - Retorno esperado del portafolio.
    """
    media_anual = np.squeeze(np.asarray(media_anual))
    return np.dot(pesos, media_anual)

def calcular_desviacion_estandar(pesos, cov_anual):
    """
    Calcula la desviación estándar (volatilidad) del portafolio en porcentaje.

    Parámetros:
    - pesos: Vector de pesos del portafolio.
    - cov_anual: Matriz de covarianza de los activos.

    Retorna:
    - La desviación estándar del portafolio en porcentaje.
    """
    # Calcular la desviación estándar del portafolio
    desviacion_estandar = np.sqrt(np.dot(pesos.T, np.dot(cov_anual, pesos)))
    
    # Convertir a porcentaje (multiplicando por 100)
    return desviacion_estandar * 100


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


def minimizar2(esperado, media_anual, cov_anual, activos):
    """
    Minimiza la varianza del portafolio con restricciones específicas para activos.

    Parámetros:
    - esperado: Retorno esperado del portafolio.
    - media_anual: Vector de rendimientos esperados de los activos.
    - cov_anual: Matriz de covarianza anual de los activos.
    - activos: Diccionario con restricciones específicas para activos.
               Formato: {indice: (cota_inferior, cota_superior)}.
    """
    media_anual = np.squeeze(np.asarray(media_anual))

    # Función objetivo: minimizar la varianza del portafolio
    def objetivo(pesos):
        return np.dot(pesos.T, np.dot(cov_anual, pesos))

    # Restricción: el retorno esperado del portafolio debe ser igual al objetivo
    def rest_esperado(pesos):
        return np.dot(pesos, media_anual) - esperado

    # Restricción: la suma de los pesos debe ser igual a 1
    def rest_suma(pesos):
        return np.sum(pesos) - 1

    # Crear lista de restricciones
    restricciones = [
        {'type': 'eq', 'fun': rest_esperado},
        {'type': 'eq', 'fun': rest_suma}
    ]

    # Agregar restricciones para cada activo
    limites = []
    for indice, (cota_inf, cota_sup) in activos.items():
        if cota_inf is None:
            cota_inf = -np.inf  # Sin límite inferior
        if cota_sup is None:
            cota_sup = np.inf  # Sin límite superior
        limites.append((cota_inf, cota_sup))

    # Pesos iniciales (distribución uniforme)
    iniciales = np.array([1 / len(media_anual)] * len(media_anual))

    # Optimización con límites en lugar de restricciones lambda
    resultado = minimize(
        objetivo,
        iniciales,
        method='SLSQP',
        constraints=restricciones,
        bounds=limites,  # Aplicar directamente las restricciones de los activos
        options={
            'disp': True,
            'ftol': 1e-9
        }
    )

    if not resultado.success:
        raise ValueError(f"Optimización fallida: {resultado.message}")

    
    # Redondear pesos a 4 decimales
    pesos_redondeados = np.round(resultado.x, 4)

    # Normalizar para que sumen exactamente 1
    pesos_normalizados = pesos_redondeados / np.sum(pesos_redondeados)


    return pesos_normalizados

# %% Montecarlo

def simulacion_portafolio_montecarlo(pesos, retornos, S0, covarianza, num_simulaciones=1000, num_periodos=252, inversion_inicial=100000):
    """
    Simula el rendimiento de un portafolio usando Monte Carlo y a partir del movimiento browniano geométrico.
    Los primeros activos son determinísticos y su rendimiento se calcula directamente.
    
    Parametros:
    - pesos: array con los pesos de cada activo en el portafolio.
    - retornos: DataFrame o array con los retornos esperados de los activos.
    - S0: array con los precios iniciales de los activos.
    - covarianza: matriz de covarianza de los activos.
    - num_simulaciones: número de simulaciones a realizar.
    - num_periodos: número de períodos (días de negociación en el año, generalmente 252).
    - inversion_inicial: valor inicial de la inversión (por ejemplo, 1,000,000).
    
    Retorna:
    - portafolio:  matriz con los valores del portafolio
    - retornos_simulados: array con los retornos simulados del portafolio al final del año.
    - retornos_logaritmicos: matriz con los retornos logarítmicos diarios del portafolio.
    """
    random.seed(123)
    # Ajuste de parámetros para la escala temporal (diaria en este caso)
    retornos_diarios = retornos / num_periodos
    covarianza_diaria = covarianza / num_periodos
    
    # Valor inicial del portafolio (invertimos una cantidad inicial en cada activo según sus pesos)
    valor_inicial_portafolio = inversion_inicial

    # Descomposición de Cholesky para generar variables correlacionadas solo para activos estocásticos
    L = np.linalg.cholesky(covarianza_diaria.iloc[2:, 2:].values)  # Solo para activos estocásticos, los primeros dos activos no se simulan
    
    num_activos = len(pesos)
    
    # Inicialización de la matriz para almacenar los valores del portafolio
    portafolio = np.zeros((num_periodos, num_simulaciones))  # Matriz para el valor del portafolio en cada simulación y periodo
    portafolio[0] = valor_inicial_portafolio  # Establecemos el valor inicial
    retornos_diarios_sim = np.zeros((num_periodos-1, num_simulaciones))
    
    # Simulaciones de Monte Carlo
    for sim in range(num_simulaciones):
        precios = np.copy(S0)  # Precios iniciales
        precio_anterior = np.copy(precios)
        
        for t in range(1, num_periodos):
            # Para los primeros dos activos determinísticos, actualizamos sus precios con sus retornos
            precios[0] *= np.exp(retornos_diarios[0])  # Primer activo determinístico
            precios[1] *= np.exp(retornos_diarios[1])  # Segundo activo determinístico
            
            # Para los activos estocásticos (de la tercera posición en adelante), aplicamos el movimiento browniano
            z = np.random.normal(0, 1, num_activos - 2)  # Ruido aleatorio solo para los activos estocásticos
            variacion = retornos_diarios[2:] - 0.5 * np.diag(covarianza_diaria.iloc[2:, 2:].values)  # Actualizamos la variación para los activos estocásticos
            ruido_correlacionado = z @ L.T
            precios[2:] *= np.exp(variacion + ruido_correlacionado)
            
            retorno_dia = np.dot(pesos, (precios / precio_anterior - 1))  # Retorno ponderado del portafolio
            precio_anterior = np.copy(precios)
            portafolio[t, sim] = portafolio[t-1, sim] * (1+ retorno_dia)

        # Calculando los retornos logaritmicos diarios para cada simulación
        retornos_diarios_sim[:, sim] = np.diff(np.log(portafolio[:, sim]))

    # Calculando los retornos anuales a partir del primer y último valor del portafolio
    retornos_anuales = np.log(portafolio[-1, :] / inversion_inicial)

    
    return portafolio, retornos_diarios_sim, retornos_anuales


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


def desviacion_downside(rendimientos_activos, pesos, rendimiento_objetivo=0):
    """
    Calcula la desviación downside anualizada del portafolio.
    
    Parámetros:
    rendimientos_activos: array-like
        Matriz de rendimientos diarios de los activos (cada columna es un activo).
    pesos: array-like
        Pesos de los activos en el portafolio. Deben sumar aproximadamente 1.
    rendimiento_objetivo: float, opcional
        Rendimiento objetivo diario (por defecto es 0).
        
    Retorna:
    float
        La desviación downside anualizada del portafolio.
    """
    primera_columna = np.full((rendimientos_activos.shape[0], 1), 0.043 / 12)
    segunda_columna = np.zeros((rendimientos_activos.shape[0], 1))
    
    # Agregar ambas columnas al inicio de la matriz
    rendimientos_activos= np.hstack((primera_columna, segunda_columna, rendimientos_activos))

    # Calcular los rendimientos diarios del portafolio
    rendimientos_portafolio = np.dot(rendimientos_activos, pesos)
    
    # Filtrar los rendimientos negativos respecto al rendimiento objetivo
    rendimientos_negativos = np.minimum(0, rendimientos_portafolio - rendimiento_objetivo)
    
    # Calcular la varianza downside diaria del portafolio
    varianza_downside_diaria = np.mean(rendimientos_negativos**2)
    
    # Desviación downside diaria
    desviacion_downside_diaria = np.sqrt(varianza_downside_diaria)
    
    # Anualizar la desviación downside
    desviacion_downside_anual = desviacion_downside_diaria * np.sqrt(252)
    
    return desviacion_downside_anual




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
    r_mercado = r_mercado.head(3494)
    r_diario = calcular_rendimientos_diarios(etf_data)
    r_diario = r_diario.head(3494)
    
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
etfs = ['XLK', 'XLV', 'XLF', 'VNQ', 'VOO', 'IBB', 'SMH', 'GLD', 'XLY', 'IYE', 'AIA', 'XRT']
start_date = '2011-01-01'

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
esp_cons = 0.06
activos_cons = {0: (0, None), # La tasa libre sin cota
           1: (0, 0), # No permitir apalancamiento
           2: (0, 0.1), # Limitar el resto de activos a no más de un 10%
           3: (0, 0.1),
           4: (0, 0.1),
           5: (0, 0.1),
           6: (0, 0.1),
           7: (0, 0.1),
           8: (0, 0.1),
           9: (0, 0.1),
           10: (0, 0.1),
           11: (0, 0.1),
           12: (0, 0.1),
           13: (0, 0.1)} 
pesos_cons = minimizar2(esp_cons, media_anual, cov_anual, activos_cons)
print("\nPesos que optimizan un portafolio conservador:")
print(pesos_cons)

# Portafolio Moderado 
esp_mod = 0.08
activos_mod = {0: (0, None), # La tasa libre sin cota
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
           11: (0, 0.2),
           12: (0, 0.2),
           13: (0, 0.2)} 
pesos_mod = minimizar2(esp_mod, media_anual, cov_anual, activos_mod)
print("\nPesos que optimizan un portafolio moderado:")
print(pesos_mod)



# Portafolio agresivo
esp_agr = 0.12
activos_agr = {0: (0, None), # La tasa libre sin cota
           1: (-0.8, 0),
           2: (-0.15, 0.3),
           3: (-0.15, 0.3),
           4: (-0.15, 0.3),
           5: (-0.15, 0.3),
           6: (-0.15, 0.3),
           7: (-0.15, 0.3),
           8: (-0.15, 0.3),
           9: (-0.15, 0.3),
           10: (-0.15, 0.3),
           11: (-0.15, 0.3),
           12: (-0.15, 0.3),
           13: (-0.15, 0.3)} 
           
pesos_agr = minimizar2(esp_agr, media_anual, cov_anual, activos_agr)
print("\nPesos que optimizan un portafolio agresivo:")
print(pesos_agr)



# Portafolio conservador
desv_1 = calcular_desviacion_estandar(pesos_cons, cov_anual)
media_1 = calcular_retorno_esperado_portafolio(pesos_cons, media_anual)

# Portafolio moderado
desv_2 = calcular_desviacion_estandar(pesos_mod, cov_anual)
media_2 = calcular_retorno_esperado_portafolio(pesos_mod, media_anual)

# Portafolio agresivo
desv_3 = calcular_desviacion_estandar(pesos_agr, cov_anual)
media_3 = calcular_retorno_esperado_portafolio(pesos_agr, media_anual)

# Imprimir los resultados
print("Resultados del Portafolio Conservador:")
print(f"Desviación Estándar: {desv_1:.2f}%")
print(f"Retorno Esperado: {media_1}\n")

print("Resultados del Portafolio Moderado:")
print(f"Desviación Estándar: {desv_2:.2f}%")
print(f"Retorno Esperado: {media_2}\n")

print("Resultados del Portafolio Agresivo:")
print(f"Desviación Estándar: {desv_3:.2f}%")
print(f"Retorno Esperado: {media_3}")

# Obtener la última observación de los precios 
S0 = etf_data.iloc[-2].values  
S0 = np.insert(S0, 0, [100000, 100000])
print(S0)


simulacion_cons = simulacion_portafolio_montecarlo(pesos_cons, media_anual.values[0], S0, cov_anual)


simulacion_mod = simulacion_portafolio_montecarlo(pesos_mod, media_anual.values[0], S0, cov_anual)


simulacion_agr = simulacion_portafolio_montecarlo(pesos_agr, media_anual.values[0], S0, cov_anual)


portafolio_simulado_mod = simulacion_mod[0]
portafolio_simulado_cons = simulacion_cons[0]
portafolio_simulado_agr = simulacion_agr[0]

retornos_anuales_cons = simulacion_cons[2]
retornos_anuales_mod = simulacion_mod[2]
retornos_anuales_agr = simulacion_agr[2]


#GRÁFICOS DE DISTRIBUCIONES DE RETORNOS ANUALES PARA CADA PORTAFOLIO


plt.figure(figsize=(10, 6))
sns.histplot(retornos_anuales_cons, kde=True, color='blue', bins=30)  # Histograma con KDE

# Etiquetas y título
plt.title('Distribución de retornos anuales para portafolio conservador ', fontsize=16)
plt.xlabel('Retorno Anual Logarítmico', fontsize=14)
plt.ylabel('Frecuencia', fontsize=14)

# Mostrar el gráfico
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(retornos_anuales_mod, kde=True, color='blue', bins=30)  # Histograma con KDE

# Etiquetas y título
plt.title('Distribución de retornos anuales para portafolio moderado ', fontsize=16)
plt.xlabel('Retorno Anual Logarítmico', fontsize=14)
plt.ylabel('Frecuencia', fontsize=14)

# Mostrar el gráfico
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(retornos_anuales_agr, kde=True, color='blue', bins=30)  # Histograma con KDE

# Etiquetas y título
plt.title('Distribución de retornos anuales para portafolio agresivo ', fontsize=16)
plt.xlabel('Retorno Anual Logarítmico', fontsize=14)
plt.ylabel('Frecuencia', fontsize=14)

# Mostrar el gráfico
plt.show()


#GRÁFICOS DE EVOLUCIÓN DE PORTAFOLIO POR PERCENTILES

# Extraemos el valor final de cada simulación (última fila)
valores_finales = portafolio_simulado_cons[-1]

# Calculamos los percentiles de los valores finales
percentil_005 = np.percentile(valores_finales, 5)
percentil_05 = np.percentile(valores_finales, 50)
percentil_095 = np.percentile(valores_finales, 95)

# Extraemos el índice de la simulación más cercana a cada percentil
indice_percentil_005 = np.argmin(np.abs(valores_finales - percentil_005))
indice_percentil_05 = np.argmin(np.abs(valores_finales - percentil_05))
indice_percentil_095 = np.argmin(np.abs(valores_finales - percentil_095))

# Extraemos las simulaciones correspondientes
simulacion_percentil_005 = portafolio_simulado_cons[:, indice_percentil_005]
simulacion_percentil_05 = portafolio_simulado_cons[:, indice_percentil_05]
simulacion_percentil_095 = portafolio_simulado_cons[:, indice_percentil_095]

# Graficamos la evolución de las simulaciones seleccionadas
plt.figure(figsize=(10, 6))

plt.plot(simulacion_percentil_005, label='Simulación Percentil 5%', color='red', linestyle='--', linewidth=2)
plt.plot(simulacion_percentil_05, label='Simulación Percentil 50%', color='blue', linewidth=2)
plt.plot(simulacion_percentil_095, label='Simulación Percentil 95%', color='green', linestyle='--', linewidth=2)

# Etiquetas y título
plt.title('Evolución del portafolio conservador (Percentiles 5%, 50%, 95%)', fontsize=16)
plt.xlabel('Tiempo (Días)', fontsize=14)
plt.ylabel('Valor del Portafolio', fontsize=14)

# Leyenda
plt.legend()

# Mostrar el gráfico
plt.show()

# Extraemos el valor final de cada simulación (última fila)
valores_finales = portafolio_simulado_mod[-1]

# Calculamos los percentiles de los valores finales
percentil_005 = np.percentile(valores_finales, 5)
percentil_05 = np.percentile(valores_finales, 50)
percentil_095 = np.percentile(valores_finales, 95)

# Extraemos el índice de la simulación más cercana a cada percentil
indice_percentil_005 = np.argmin(np.abs(valores_finales - percentil_005))
indice_percentil_05 = np.argmin(np.abs(valores_finales - percentil_05))
indice_percentil_095 = np.argmin(np.abs(valores_finales - percentil_095))

# Extraemos las simulaciones correspondientes
simulacion_percentil_005 = portafolio_simulado_mod[:, indice_percentil_005]
simulacion_percentil_05 = portafolio_simulado_mod[:, indice_percentil_05]
simulacion_percentil_095 = portafolio_simulado_mod[:, indice_percentil_095]

# Graficamos la evolución de las simulaciones seleccionadas
plt.figure(figsize=(10, 6))

plt.plot(simulacion_percentil_005, label='Simulación Percentil 5%', color='red', linestyle='--', linewidth=2)
plt.plot(simulacion_percentil_05, label='Simulación Percentil 50%', color='blue', linewidth=2)
plt.plot(simulacion_percentil_095, label='Simulación Percentil 95%', color='green', linestyle='--', linewidth=2)

# Etiquetas y título
plt.title('Evolución del portafolio moderado (Percentiles 5%, 50%, 95%)', fontsize=16)
plt.xlabel('Tiempo (Días)', fontsize=14)
plt.ylabel('Valor del Portafolio', fontsize=14)

# Leyenda
plt.legend()

# Mostrar el gráfico
plt.show()


# Extraemos el valor final de cada simulación (última fila)
valores_finales = portafolio_simulado_agr[-1]

# Calculamos los percentiles de los valores finales
percentil_005 = np.percentile(valores_finales, 5)
percentil_05 = np.percentile(valores_finales, 50)
percentil_095 = np.percentile(valores_finales, 95)

# Extraemos el índice de la simulación más cercana a cada percentil
indice_percentil_005 = np.argmin(np.abs(valores_finales - percentil_005))
indice_percentil_05 = np.argmin(np.abs(valores_finales - percentil_05))
indice_percentil_095 = np.argmin(np.abs(valores_finales - percentil_095))

# Extraemos las simulaciones correspondientes
simulacion_percentil_005 = portafolio_simulado_agr[:, indice_percentil_005]
simulacion_percentil_05 = portafolio_simulado_agr[:, indice_percentil_05]
simulacion_percentil_095 = portafolio_simulado_agr[:, indice_percentil_095]

# Graficamos la evolución de las simulaciones seleccionadas
plt.figure(figsize=(10, 6))

plt.plot(simulacion_percentil_005, label='Simulación Percentil 5%', color='red', linestyle='--', linewidth=2)
plt.plot(simulacion_percentil_05, label='Simulación Percentil 50%', color='blue', linewidth=2)
plt.plot(simulacion_percentil_095, label='Simulación Percentil 95%', color='green', linestyle='--', linewidth=2)

# Etiquetas y título
plt.title('Evolución del portafolio agresivo (Percentiles 5%, 50%, 95%)', fontsize=16)
plt.xlabel('Tiempo (Días)', fontsize=14)
plt.ylabel('Valor del Portafolio', fontsize=14)

# Leyenda
plt.legend()

# Mostrar el gráfico
plt.show()



# Calculo de Capm y betas
#result_capm = capm(etfs, start_date, 0.05, pesos_min_2)
#print("\nResultados del Capm")
#print(result_capm)


#MÉTRICAS

#Sharpe Ratio
sharpe_con = sharpe_ratio(media_1, 0.043, desv_1)
sharpe_mod = sharpe_ratio(media_2, 0.043, desv_2)
sharpe_agr = sharpe_ratio(media_3, 0.043, desv_3)

#Roy's safety first ratio
#el mínimo de rendimiento es 1% menos del esperado
roy_con = safety_first_ratio(media_1, 0.05, desv_1)
roy_mod = safety_first_ratio(media_2, 0.07, desv_2)
roy_agr = safety_first_ratio(media_3, 0.011, desv_3)

#Sortino Radio
#Calculo de la desviacion downside con tasa objetivo la libre de riesgo
desv_down_1 = desviacion_downside(r_diario, pesos_cons, 0.048/12)
desv_down_2 = desviacion_downside(r_diario, pesos_mod, 0.048/12)
desv_down_3 = desviacion_downside(r_diario, pesos_agr, 0.048/12)


sortino_con = sortino_ratio(media_1, 0.043, desv_down_1)
sortino_mod = sortino_ratio(media_2, 0.043, desv_down_1)
sortino_agr = sortino_ratio(media_3, 0.043, desv_down_1)

#Treynor Ratio

capm_1 = capm(etfs, start_date, 0.043, pesos_cons)
capm_2 = capm(etfs, start_date, 0.043, pesos_cons)
capm_3 = capm(etfs, start_date, 0.043, pesos_cons)


treynor_con = treynor_ratio(media_1, 0.043, capm_1['beta_port'])
treynor_mod = treynor_ratio(media_2, 0.043, capm_1['beta_port'])
treynor_agr = treynor_ratio(media_3, 0.043, capm_1['beta_port'])

#Jensen's alpha

jensen_con = jensens_alpha(media_1, etfs, start_date, 0.043, pesos_cons)
jensen_mod = jensens_alpha(media_2, etfs, start_date, 0.043, pesos_mod)
jensen_agr = jensens_alpha(media_3, etfs, start_date, 0.043, pesos_agr)























