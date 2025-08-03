import requests
import json
import argparse

def test_deployed_api(api_url, lat=None, lon=None):
    """
    Prueba la API desplegada en Render
    
    Args:
        api_url: URL base de la API desplegada (sin / al final)
        lat: Latitud opcional para la prueba
        lon: Longitud opcional para la prueba
    """
    # Endpoint a probar
    forecast_url = f"{api_url}/recent_forecast"
    
    # Datos de prueba - usar argumentos o valores por defecto
    if lat is None or lon is None:
        # Coordenadas por defecto (Cali, Colombia)
        payload = {
            "lat": 3.4,
            "lon": -76.5
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
        # Verificar si la API está en línea
        print(f"Verificando si la API está en línea en {api_url}...")
        home_response = requests.get(api_url)
        if home_response.status_code == 200:
            print("✅ API en línea!\n")
        else:
            print(f"⚠️ La página principal respondió con código {home_response.status_code}\n")
        
        # Hacer la solicitud POST al endpoint de pronóstico
        print(f"Enviando solicitud a {forecast_url} con coordenadas: {payload}...")
        response = requests.post(forecast_url, json=payload, headers=headers, timeout=30)
        
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

if __name__ == "__main__":
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(description='Probar la API desplegada en Render.')
    parser.add_argument('--url', type=str, required=True, help='URL base de la API desplegada (ej: https://siria-drought-api.onrender.com)')
    parser.add_argument('--lat', type=float, help='Latitud para la prueba')
    parser.add_argument('--lon', type=float, help='Longitud para la prueba')
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Llamar a la función de prueba con los argumentos proporcionados
    test_deployed_api(args.url, args.lat, args.lon)
