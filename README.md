# API para Predicción de Humedad del Suelo con ERA5-Land

Esta API permite realizar predicciones de la humedad del suelo (soil_water_1) utilizando un modelo MLP entrenado con datos de ERA5-Land para América Latina. Además, proporciona alertas sobre el estado del suelo basadas en datos recientes de Copernicus.

## Requisitos

- Python 3.8 o superior
- Dependencias listadas en `requirements.txt`
- Credenciales de Copernicus Climate Data Store (CDS)

## Instalación

1. Asegúrate de tener Python instalado.
2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

3. Asegúrate de que los archivos del modelo y el scaler estén presentes en el directorio:
   - `drought_mlp_model.pkl`
   - `minmax_scaler.pkl`

4. Configura las variables de entorno:
   - Copia `.env.example` a `.env`
   - Actualiza las variables con tus credenciales y configuración
   - Asegúrate de incluir tu clave API de Copernicus CDS en `CDS_KEY`

## Ejecución

Para ejecutar la API localmente:

```bash
python app.py
```

Por defecto, la API se ejecutará en `http://localhost:5000/`

## Endpoints Disponibles

### 1. Página Principal
- **URL**: `/`
- **Método**: GET
- **Descripción**: Muestra información general sobre la API.

### 2. Predicción Individual
- **URL**: `/predict`
- **Método**: POST
- **Formato de Entrada**:
```json
{
    "soil_water_2": 0.4,
    "soil_water_3": 0.35,
    "soil_water_4": 0.3,
    "soil_temp_lvl1": 0.7,
    "lat": -34.6,
    "lon": -58.4,
    "pais": "Argentina"
}
```
- **Respuesta Exitosa**:
```json
{
    "prediction": 0.45,
    "message": "Predicción de humedad del suelo (soil_water_1) para Argentina: 0.4500"
}
```

### 3. Predicción en Lote
- **URL**: `/predict_batch`
- **Método**: POST
- **Formato de Entrada**:
```json
{
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
```
- **Respuesta Exitosa**:
```json
{
    "results": [
        {
            "index": 0,
            "pais": "Argentina",
            "prediction": 0.45,
            "coordinates": [-34.6, -58.4]
        }
    ]
}
```

### 4. Información del Modelo
- **URL**: `/model_info`
- **Método**: GET
- **Descripción**: Devuelve información sobre el modelo MLP entrenado.

### 5. Pronóstico con Datos Recientes
- **URL**: `/recent_forecast`
- **Método**: POST
- **Formato de Entrada**:
```json
{
    "lat": 3.4,
    "lon": -76.5
}
```
- **Respuesta Exitosa**:
```json
{
    "coordinates": {
        "requested": {"lat": 3.4, "lon": -76.5},
        "reference_point": {
            "lat": 3.4, 
            "lon": -76.5,
            "description": "Cali",
            "distance_km": 0.0
        }
    },
    "country": "Colombia",
    "period": {
        "start_date": "2025-07-26",
        "end_date": "2025-07-31"
    },
    "soil_moisture_prediction": {
        "soil_water_1": 0.5858,
        "message": "Predicción de humedad del suelo (capa 1): 0.5858"
    },
    "soil_condition": {
        "condition": "Húmedo",
        "alert_level": "Baja",
        "message": "ALERTA BAJA: Suelo con buena humedad. Considere reducir ligeramente el riego."
    },
    "stats": {
        "avg_soil_moisture": 0.4061,
        "avg_precipitation": 0.0029,
        "max_precipitation": 0.0047,
        "soil_moisture_trend": 0.0293
    }
}
```,
        {
            "index": 1,
            "pais": "Venezuela",
            "prediction": 0.52,
            "coordinates": [10.5, -66.9]
        }
    ]
}
```

### 4. Información del Modelo
- **URL**: `/model_info`
- **Método**: GET
- **Descripción**: Muestra información técnica sobre el modelo MLP.

## Ejemplos de Uso con cURL

### Predicción Individual
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "soil_water_2": 0.4,
    "soil_water_3": 0.35,
    "soil_water_4": 0.3,
    "soil_temp_lvl1": 0.7,
    "lat": -34.6,
    "lon": -58.4,
    "pais": "Argentina"
  }'
```

### Predicción en Lote
```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Obtener Pronóstico con Datos Recientes
```bash
curl -X POST http://localhost:5000/recent_forecast \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 3.4,
    "lon": -76.5
  }'
```

## Scripts de Prueba

- **`test_recent_forecast.py`**: Prueba básica del endpoint de pronóstico
- **`test_forecast_env.py`**: Versión mejorada que utiliza variables de entorno y permite especificar coordenadas

Ejemplo de uso con coordenadas personalizadas:
```bash
python test_forecast_env.py --lat 19.4 --lon -99.1
```

## Variables de Entorno

La API utiliza variables de entorno para su configuración. Estas pueden definirse en un archivo `.env`:

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `FLASK_APP` | Nombre del archivo principal | `app.py` |
| `FLASK_ENV` | Entorno de ejecución | `development` |
| `FLASK_DEBUG` | Modo de depuración | `1` (activado) |
| `PORT` | Puerto para la API | `5000` |
| `MODEL_PATH` | Ruta al modelo | `drought_mlp_model.pkl` |
| `SCALER_PATH` | Ruta al scaler | `minmax_scaler.pkl` |
| `CDS_URL` | URL de Copernicus CDS API | `https://cds.climate.copernicus.eu/api/v2` |
| `CDS_KEY` | Clave API de Copernicus CDS | - |
| `CDS_DATASET` | Dataset de Copernicus | `reanalysis-era5-land` |

## Notas Importantes

- Las variables numéricas `soil_water_2`, `soil_water_3`, `soil_water_4` y `soil_temp_lvl1` deben estar normalizadas (entre 0 y 1).
- Solo se aceptan países de América Latina que estén en la lista predefinida.
- Las coordenadas (lat, lon) deben corresponder a puntos en América Latina.
- Para usar el endpoint `/recent_forecast`, se requieren credenciales de Copernicus CDS.
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Información del Modelo
```bash
curl http://localhost:5000/model_info
```
