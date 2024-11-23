# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 01:24:00 2024

@author: AMADOR
"""

import yfinance as yf
import pandas as pd
import os

# Definir los símbolos de los ETFs
etfs = ['XLK', 'XLV', 'XLF', 'EFA', 'VNQ', 'ICLN', 'JNK', 'LQD', 'DBC', 'VWO']

# Descargar los datos históricos de los ETFs
etf_data10 = yf.download(etfs, start='2000-01-01')

# Obtener la ruta relativa para la carpeta 'data' en el directorio padre
ruta_carpeta_data = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
#os.makedirs(ruta_carpeta_data, exist_ok=True)  

ruta_csv = os.path.join(ruta_carpeta_data, 'etf_data10.csv')

# Guardar los datos en un archivo CSV
etf_data10.to_csv(ruta_csv)

# Mostrar
print(etf_data10.head())

# Mostrar los últimos
print(etf_data10.tail())
