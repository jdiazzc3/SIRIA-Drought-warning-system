#!/usr/bin/env bash
# build.sh para Render.com

echo "Actualizando pip, setuptools y wheel..."
pip install --upgrade pip setuptools wheel

echo "Instalando werkzeug explícitamente (dependencia de Flask)..."
pip install werkzeug==2.3.7

echo "Instalando paquetes científicos usando wheels pre-compilados..."
pip install --only-binary=:all: numpy pandas scikit-learn xarray netCDF4

echo "Instalando Flask con sus dependencias..."
pip install Flask==2.3.3

echo "Instalando resto de dependencias..."
pip install cdsapi python-dotenv requests gunicorn

echo "Verificando las versiones instaladas..."
pip list
