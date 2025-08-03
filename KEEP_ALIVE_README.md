# Mantener el servicio activo en Render

Este proyecto incluye configuraciones para mantener el servicio activo en Render evitando que entre en modo de suspensión por inactividad. Se implementan dos soluciones complementarias:

## 1. Endpoint `/ping` en la API Flask

El archivo `app.py` incluye un endpoint ligero `/ping` que devuelve un simple JSON con información de estado y timestamp. Este endpoint está diseñado específicamente para recibir "pings" y mantener el servicio activo.

```python
@app.route('/ping', methods=['GET'])
def ping():
    """Endpoint liviano para mantener el servicio activo"""
    return jsonify({
        "status": "alive",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "service": "drought-api"
    })
```

## 2. GitHub Actions para ping periódico

Se incluye un flujo de trabajo de GitHub Actions en `.github/workflows/keep_alive.yml` que realiza solicitudes periódicas al endpoint `/ping` para mantener el servicio activo.

### Instrucciones para configurar GitHub Actions:

1. **Actualizar la URL del servicio**: Modifica el archivo `.github/workflows/keep_alive.yml` y reemplaza `https://tu-servicio-en-render.onrender.com` con la URL real de tu servicio en Render.

2. **Habilitar GitHub Actions**:
   - Sube todo el código a tu repositorio de GitHub
   - Ve a la pestaña "Actions" en tu repositorio
   - Habilita los flujos de trabajo si no están habilitados
   - Verifica que el flujo de trabajo "Keep Render Service Alive" aparezca en la lista

3. **Ejecutar manualmente** (opcional):
   - En la pestaña "Actions", selecciona "Keep Render Service Alive"
   - Haz clic en "Run workflow" para ejecutar manualmente el flujo de trabajo y verificar que funcione correctamente

## Notas importantes:

- El flujo de trabajo está configurado para ejecutarse cada 10 minutos. Puedes ajustar esto modificando la expresión cron en el archivo YAML.
- GitHub Actions tiene límites de uso en su plan gratuito, pero este flujo de trabajo es muy ligero y no debería causar problemas.
- Si tu servicio en Render tiene un tiempo de inactividad menor a 10 minutos, considera ajustar la frecuencia de ejecución del flujo de trabajo.

## Verificación:

Para verificar que el sistema está funcionando correctamente, puedes:

1. Acceder directamente a `https://tu-servicio-en-render.onrender.com/ping` en tu navegador
2. Revisar los registros de ejecución del flujo de trabajo en GitHub Actions
3. Comprobar en el panel de Render que tu servicio permanece activo
