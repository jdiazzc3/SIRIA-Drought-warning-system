import requests
import json

def test_recent_forecast():
    """Prueba el endpoint de pronóstico reciente"""
    url = "http://localhost:5000/recent_forecast"
    
    # Datos de prueba para Buenos Aires, Argentina
    payload = {
        "lat": 3.388909268582466,
        "lon": -76.53779766459249
    }
# 3.388909268582466, -76.53779766459249
    
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
        else:
            print(f"Error: {response.text}")
    
    except Exception as e:
        print(f"Error en la solicitud: {e}")

# Punto de entrada
if __name__ == "__main__":
    test_recent_forecast()
