#!/usr/bin/env bash
# build.sh para Render.com

echo "Actualizando pip, setuptools y wheel..."
pip install --upgrade pip setuptools wheel

echo "Instalando paquetes científicos usando wheels pre-compilados..."
pip install --only-binary=:all: numpy pandas scikit-learn xarray netCDF4

echo "Instalando resto de dependencias..."
pip install --no-deps Flask cdsapi python-dotenv requests gunicorn

echo "Verificando las versiones instaladas..."
pip list
