import requests
import json
import pandas as pd
import numpy as np
import os

# Verificar si los archivos del modelo existen
model_path = 'drought_mlp_model.pkl'
scaler_path = 'minmax_scaler.pkl'

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    print(f"Error: No se encontraron los archivos del modelo o el scaler.")
    print(f"Directorio actual: {os.getcwd()}")
    print(f"Archivos en el directorio: {os.listdir()}")
    exit(1)

# URL de la API (asumiendo que se ejecuta localmente)
API_URL = "http://localhost:5000"

def test_home():
    """Prueba el endpoint principal"""
    print("\n1. Probando el endpoint principal (/)")
    response = requests.get(f"{API_URL}/")
    
    if response.status_code == 200:
        print("✓ Endpoint principal funciona correctamente")
    else:
        print(f"✗ Error en el endpoint principal: {response.status_code}")
        print(response.text)

def test_model_info():
    """Prueba el endpoint de información del modelo"""
    print("\n2. Probando el endpoint de información del modelo (/model_info)")
    response = requests.get(f"{API_URL}/model_info")
    
    if response.status_code == 200:
        model_info = response.json()
        print("✓ Información del modelo obtenida correctamente")
        print(f"   - Tipo de modelo: {model_info.get('model_type')}")
        print(f"   - Arquitectura: {model_info.get('hidden_layer_sizes')}")
        print(f"   - Activación: {model_info.get('activation')}")
        print(f"   - Optimizador: {model_info.get('solver')}")
    else:
        print(f"✗ Error al obtener información del modelo: {response.status_code}")
        print(response.text)

def test_prediction():
    """Prueba el endpoint de predicción individual"""
    print("\n3. Probando el endpoint de predicción individual (/predict)")
    
    # Datos de ejemplo para la predicción
    test_data = {
        "soil_water_2": 0.4,
        "soil_water_3": 0.35,
        "soil_water_4": 0.3,
        "soil_temp_lvl1": 0.7,
        "lat": -34.6,
        "lon": -58.4,
        "pais": "Argentina"
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data)
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Predicción individual realizada correctamente")
        print(f"   - Predicción: {result.get('prediction')}")
        print(f"   - Mensaje: {result.get('message')}")
    else:
        print(f"✗ Error en la predicción individual: {response.status_code}")
        print(response.text)

def test_batch_prediction():
    """Prueba el endpoint de predicción en lote"""
    print("\n4. Probando el endpoint de predicción en lote (/predict_batch)")
    
    # Datos de ejemplo para la predicción en lote
    test_batch_data = {
        "data": [
            {
                "soil_water_2": 0.4,
                "soil_water_3": 0.35,
                "soil_water_4": 0.3,
                "soil_temp_lvl1": 0.7,
                "lat": -34.6,
                "lon": -58.4,
                "pais": "Argentina"
            },
            {
                "soil_water_2": 0.5,
                "soil_water_3": 0.45,
                "soil_water_4": 0.4,
                "soil_temp_lvl1": 0.8,
                "lat": 10.5,
                "lon": -66.9,
                "pais": "Venezuela"
            }
        ]
    }
    
    response = requests.post(
        f"{API_URL}/predict_batch",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_batch_data)
    )
    
    if response.status_code == 200:
        results = response.json()
        print("✓ Predicción en lote realizada correctamente")
        for i, result in enumerate(results.get('results', [])):
            print(f"   - Resultado {i+1}: País={result.get('pais')}, Predicción={result.get('prediction')}")
    else:
        print(f"✗ Error en la predicción en lote: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("=== PRUEBA DE LA API DE PREDICCIÓN DE HUMEDAD DEL SUELO ===")
    
    try:
        # Ejecutar todas las pruebas
        test_home()
        test_model_info()
        test_prediction()
        test_batch_prediction()
        
        print("\n✓ Todas las pruebas completadas")
    except requests.exceptions.ConnectionError:
        print("\n✗ Error de conexión: No se pudo conectar a la API.")
        print("  Asegúrate de que la API esté en ejecución en http://localhost:5000")
    except Exception as e:
        print(f"\n✗ Error durante las pruebas: {str(e)}")
