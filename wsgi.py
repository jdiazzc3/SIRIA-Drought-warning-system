import os
import sys

# Asegurarse de que el directorio actual está en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar la aplicación Flask desde app.py
try:
    # Intentar importar directamente
    from app import app as application
except ModuleNotFoundError as e:
    # Si hay un error, mostrar un mensaje detallado
    print(f"Error al importar la aplicación: {e}")
    print("Entorno Python:")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path}")
    print("\nPaquetes instalados:")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "list"])
    raise

# Esta variable será usada por gunicorn
app = application

if __name__ == "__main__":
    # Ejecutar la aplicación directamente si este script es ejecutado
    application.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
