#!/usr/bin/env bash
# build.sh para Render.com

# Actualizar pip a la última versión
pip install --upgrade pip setuptools wheel

# Instalar paquetes binarios primero 
pip install --only-binary=:all: numpy pandas scikit-learn xarray netCDF4

# Instalar el resto de las dependencias
pip install -r requirements.txt
