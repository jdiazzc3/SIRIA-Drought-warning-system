import requests
import time
import argparse
from datetime import datetime

def test_ping_endpoint(url, num_pings=5, interval=2):
    """
    Prueba el endpoint de ping haciendo solicitudes periódicas
    
    Args:
        url: URL completa del endpoint de ping
        num_pings: Número de pings a realizar
        interval: Intervalo entre pings en segundos
    """
    print(f"Probando endpoint de ping: {url}")
    print(f"Realizando {num_pings} pings con {interval} segundos de intervalo\n")
    
    for i in range(num_pings):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Ping #{i+1} - {timestamp}")
            
            # Realizar solicitud GET al endpoint de ping
            start_time = time.time()
            response = requests.get(url)
            elapsed_time = time.time() - start_time
            
            # Mostrar resultados
            print(f"  Estado: {response.status_code}")
            print(f"  Tiempo de respuesta: {elapsed_time:.4f} segundos")
            
            if response.status_code == 200:
                data = response.json()
                print(f"  Respuesta: {data}")
            else:
                print(f"  Error: {response.text}")
                
            # Esperar antes del siguiente ping
            if i < num_pings - 1:
                print(f"  Esperando {interval} segundos...")
                time.sleep(interval)
                
        except Exception as e:
            print(f"  Error en la solicitud: {e}")
    
    print("\nPrueba completada.")

if __name__ == "__main__":
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Probar el endpoint de ping")
    parser.add_argument("--url", type=str, 
                        default="http://localhost:5000/ping",
                        help="URL del endpoint de ping")
    parser.add_argument("--num", type=int, default=5,
                        help="Número de pings a realizar")
    parser.add_argument("--interval", type=int, default=2,
                        help="Intervalo entre pings en segundos")
    
    args = parser.parse_args()
    
    # Ejecutar prueba
    test_ping_endpoint(args.url, args.num, args.interval)
