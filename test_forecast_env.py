import requests
import json
import os
import argparse
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

def test_recent_forecast(lat=None, lon=None):
    """Prueba el endpoint de pronóstico reciente"""
    # Configurar URL desde variable de entorno o usar valor por defecto
    port = os.getenv('PORT', 5000)
    url = f"http://localhost:{port}/recent_forecast"
    
    # Datos de prueba - usar argumentos o valores por defecto
    if lat is None or lon is None:
        # Coordenadas por defecto (Cali, Colombia)
        payload = {
            "lat": 3.388909268582466,
            "lon": -76.53779766459249
        }
    else:
        payload = {
            "lat": lat,
            "lon": lon
        }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Hacer la solicitud POST
        response = requests.post(url, json=payload, headers=headers)
        
        # Imprimir el código de estado
        print(f"Estado: {response.status_code}")
        
        # Imprimir la respuesta formateada
        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data, indent=4, ensure_ascii=False))
            
            # Mostrar detalles específicos sobre el estado del suelo
            print("\n--- RESUMEN DE ALERTA ---")
            print(f"Estado del suelo: {data['soil_condition']['condition']}")
            print(f"Nivel de alerta: {data['soil_condition']['alert_level']}")
            print(f"Mensaje: {data['soil_condition']['message']}")
            print(f"Humedad predicha: {data['soil_moisture_prediction']['soil_water_1']:.4f}")
        else:
            print(f"Error: {response.text}")
    
    except Exception as e:
        print(f"Error en la solicitud: {e}")

# Punto de entrada
if __name__ == "__main__":
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(description='Probar el endpoint de pronóstico reciente.')
    parser.add_argument('--lat', type=float, help='Latitud para la prueba')
    parser.add_argument('--lon', type=float, help='Longitud para la prueba')
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Llamar a la función de prueba con los argumentos proporcionados
    test_recent_forecast(args.lat, args.lon)
