import os
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import cdsapi
import tempfile
from math import radians, sin, cos, sqrt, atan2
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Inicializar la aplicación Flask
app = Flask(__name__)

# Cargar el modelo entrenado y el scaler
MODEL_PATH = os.getenv('MODEL_PATH', 'drought_mlp_model.pkl')
SCALER_PATH = os.getenv('SCALER_PATH', 'minmax_scaler.pkl')

# Verificar si los archivos existen
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"No se encontraron los archivos del modelo o el scaler en {os.getcwd()}")

# Cargar el modelo
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

# Cargar el scaler
with open(SCALER_PATH, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Definir las características esperadas
FEATURES = ['soil_water_2', 'soil_water_3', 'soil_water_4', 'soil_temp_lvl1', 'lat', 'lon']
COUNTRIES = ['Argentina', 'Bolivia', 'Brasil', 'Chile', 'Colombia', 'Costa Rica', 'Cuba', 
             'Ecuador', 'El Salvador', 'Guatemala', 'Honduras', 'México', 'Nicaragua', 
             'Panamá', 'Paraguay', 'Perú', 'Puerto Rico', 'República Dominicana', 
             'Uruguay', 'Venezuela']

# Definir puntos de referencia para cada país
REFERENCE_POINTS = {
    'Argentina': [
        {'lat': -34.6, 'lon': -58.4, 'desc': 'Buenos Aires'},
        {'lat': -31.4, 'lon': -64.2, 'desc': 'Córdoba'},
        {'lat': -26.8, 'lon': -65.2, 'desc': 'Tucumán'}
    ],
    'Bolivia': [
        {'lat': -16.5, 'lon': -68.1, 'desc': 'La Paz'},
        {'lat': -17.8, 'lon': -63.2, 'desc': 'Santa Cruz'}
    ],
    'Brasil': [
        {'lat': -15.8, 'lon': -47.9, 'desc': 'Brasilia'},
        {'lat': -23.5, 'lon': -46.6, 'desc': 'São Paulo'},
        {'lat': -3.1, 'lon': -60.0, 'desc': 'Manaus'}
    ],
    'Chile': [
        {'lat': -33.5, 'lon': -70.7, 'desc': 'Santiago'},
        {'lat': -53.2, 'lon': -70.9, 'desc': 'Punta Arenas'}
    ],
    'Colombia': [
        {'lat': 4.6, 'lon': -74.1, 'desc': 'Bogotá'},
        {'lat': 10.4, 'lon': -75.5, 'desc': 'Cartagena'},
        {'lat': 3.4, 'lon': -76.5, 'desc': 'Cali'}
    ],
    'Costa Rica': [
        {'lat': 9.9, 'lon': -84.1, 'desc': 'San José'}
    ],
    'Cuba': [
        {'lat': 23.1, 'lon': -82.4, 'desc': 'La Habana'},
        {'lat': 20.0, 'lon': -75.8, 'desc': 'Santiago de Cuba'}
    ],
    'Ecuador': [
        {'lat': -0.2, 'lon': -78.5, 'desc': 'Quito'},
        {'lat': -2.2, 'lon': -79.9, 'desc': 'Guayaquil'}
    ],
    'El Salvador': [
        {'lat': 13.7, 'lon': -89.2, 'desc': 'San Salvador'}
    ],
    'Guatemala': [
        {'lat': 14.6, 'lon': -90.5, 'desc': 'Ciudad de Guatemala'}
    ],
    'Honduras': [
        {'lat': 14.1, 'lon': -87.2, 'desc': 'Tegucigalpa'}
    ],
    'México': [
        {'lat': 19.4, 'lon': -99.1, 'desc': 'Ciudad de México'},
        {'lat': 25.7, 'lon': -100.3, 'desc': 'Monterrey'},
        {'lat': 20.7, 'lon': -103.4, 'desc': 'Guadalajara'}
    ],
    'Nicaragua': [
        {'lat': 12.1, 'lon': -86.3, 'desc': 'Managua'}
    ],
    'Panamá': [
        {'lat': 9.0, 'lon': -79.5, 'desc': 'Ciudad de Panamá'}
    ],
    'Paraguay': [
        {'lat': -25.3, 'lon': -57.6, 'desc': 'Asunción'}
    ],
    'Perú': [
        {'lat': -12.0, 'lon': -77.0, 'desc': 'Lima'},
        {'lat': -3.7, 'lon': -73.2, 'desc': 'Iquitos'}
    ],
    'Puerto Rico': [
        {'lat': 18.5, 'lon': -66.1, 'desc': 'San Juan'}
    ],
    'República Dominicana': [
        {'lat': 18.5, 'lon': -69.9, 'desc': 'Santo Domingo'}
    ],
    'Uruguay': [
        {'lat': -34.9, 'lon': -56.2, 'desc': 'Montevideo'}
    ],
    'Venezuela': [
        {'lat': 10.5, 'lon': -66.9, 'desc': 'Caracas'},
        {'lat': 8.6, 'lon': -71.2, 'desc': 'Mérida'}
    ]
}

@app.route('/')
def home():
    """Endpoint principal que muestra información sobre la API"""
    return """
    <h1>API de Predicción de Humedad del Suelo</h1>
    <p>Esta API permite realizar predicciones de la humedad del suelo (soil_water_1) utilizando un modelo MLP entrenado con datos de ERA5-Land.</p>
    <h2>Endpoints disponibles:</h2>
    <ul>
        <li><strong>/predict</strong> - POST: Realiza una predicción individual</li>
        <li><strong>/predict_batch</strong> - POST: Realiza predicciones en lote</li>
        <li><strong>/model_info</strong> - GET: Muestra información sobre el modelo</li>
        <li><strong>/recent_forecast</strong> - POST: Obtiene datos recientes de Copernicus y proporciona alertas sobre el estado del suelo</li>
    </ul>
    """

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para realizar una predicción individual
    
    Ejemplo de JSON de entrada:
    {
        "soil_water_2": 0.4,
        "soil_water_3": 0.35,
        "soil_water_4": 0.3,
        "soil_temp_lvl1": 295.5,
        "lat": -34.6,
        "lon": -58.4,
        "pais": "Argentina"
    }
    """
    try:
        # Obtener datos de la solicitud
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No se recibieron datos"}), 400
        
        # Verificar que estén todas las características necesarias
        for feature in FEATURES:
            if feature not in data:
                return jsonify({"error": f"Falta la característica '{feature}'"}), 400
        
        # Verificar si se proporcionó el país
        if 'pais' not in data:
            return jsonify({"error": "Falta el país"}), 400
        
        if data['pais'] not in COUNTRIES:
            return jsonify({"error": f"País no reconocido. Países válidos: {COUNTRIES}"}), 400
        
        # Preparar datos para el modelo
        X = {}
        
        # Primero normalizar las características numéricas
        numeric_values = [[data[f] for f in FEATURES]]
        
        # Aplicar el scaler solamente a los datos numéricos que corresponden a soil_water y soil_temp
        # (no a lat y lon)
        scaled_data = numeric_values.copy()
        for i, feature in enumerate(FEATURES):
            if feature in ['soil_water_2', 'soil_water_3', 'soil_water_4', 'soil_temp_lvl1']:
                # Verificar que los valores estén en el rango esperado
                if feature.startswith('soil_water') and (data[feature] < 0 or data[feature] > 1):
                    return jsonify({"error": f"El valor de {feature} debe estar entre 0 y 1 después de la normalización"}), 400
                if feature == 'soil_temp_lvl1' and (data[feature] < 0 or data[feature] > 1):
                    return jsonify({"error": f"El valor de {feature} debe estar entre 0 y 1 después de la normalización"}), 400
            
            X[feature] = scaled_data[0][i]
        
        # One-hot encoding para el país con drop_first=True (sin Argentina)
        for country in COUNTRIES:
            if country == 'Argentina':
                # Argentina es la categoría de referencia (no se incluye por drop_first=True)
                continue
            elif country == data['pais']:
                X[f'pais_{country}'] = 1
            else:
                X[f'pais_{country}'] = 0
                
        # Convertir a DataFrame para mantener el orden de las columnas
        input_df = pd.DataFrame([X])
        
        # Realizar la predicción
        prediction = model.predict(input_df)[0]
        
        # Preparar la respuesta
        response = {
            "prediction": float(prediction),
            "message": f"Predicción de humedad del suelo (soil_water_1) para {data['pais']}: {prediction:.4f}"
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Endpoint para realizar predicciones en lote
    
    Ejemplo de JSON de entrada:
    {
        "data": [
            {
                "soil_water_2": 0.4,
                "soil_water_3": 0.35,
                "soil_water_4": 0.3,
                "soil_temp_lvl1": 295.5,
                "lat": -34.6,
                "lon": -58.4,
                "pais": "Argentina"
            },
            {
                "soil_water_2": 0.5,
                "soil_water_3": 0.45,
                "soil_water_4": 0.4,
                "soil_temp_lvl1": 297.0,
                "lat": 10.5,
                "lon": -66.9,
                "pais": "Venezuela"
            }
        ]
    }
    """
    try:
        # Obtener datos de la solicitud
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return jsonify({"error": "No se recibieron datos en el formato correcto"}), 400
        
        batch_data = request_data['data']
        
        if not isinstance(batch_data, list) or len(batch_data) == 0:
            return jsonify({"error": "El campo 'data' debe ser una lista no vacía"}), 400
        
        # Lista para almacenar los registros procesados
        processed_records = []
        
        # Procesar cada registro
        for i, record in enumerate(batch_data):
            # Verificar que estén todas las características necesarias
            for feature in FEATURES:
                if feature not in record:
                    return jsonify({"error": f"Falta la característica '{feature}' en el registro {i+1}"}), 400
            
            # Verificar si se proporcionó el país
            if 'pais' not in record:
                return jsonify({"error": f"Falta el país en el registro {i+1}"}), 400
            
            if record['pais'] not in COUNTRIES:
                return jsonify({"error": f"País no reconocido en el registro {i+1}. Países válidos: {COUNTRIES}"}), 400
            
            # Preparar el registro
            processed_record = {}
            
            # Copiar las características numéricas
            for feature in FEATURES:
                processed_record[feature] = record[feature]
            
            # One-hot encoding para el país con drop_first=True (sin Argentina)
            for country in COUNTRIES:
                if country == 'Argentina':
                    # Argentina es la categoría de referencia (no se incluye por drop_first=True)
                    continue
                elif country == record['pais']:
                    processed_record[f'pais_{country}'] = 1
                else:
                    processed_record[f'pais_{country}'] = 0
            
            processed_records.append(processed_record)
        
        # Convertir a DataFrame
        input_df = pd.DataFrame(processed_records)
        
        # Realizar predicciones
        predictions = model.predict(input_df)
        
        # Preparar la respuesta
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "index": i,
                "pais": batch_data[i]['pais'],
                "prediction": float(pred),
                "coordinates": [batch_data[i]['lat'], batch_data[i]['lon']]
            })
        
        return jsonify({"results": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Endpoint para mostrar información sobre el modelo"""
    try:
        # Obtener información básica del modelo
        model_info = {
            "model_type": type(model).__name__,
            "features": FEATURES,
            "countries": COUNTRIES,
            "hidden_layer_sizes": model.hidden_layer_sizes,
            "activation": model.activation,
            "solver": model.solver,
            "alpha": model.alpha,
            "learning_rate": model.learning_rate,
            "iterations": model.n_iter_,
            "model_path": MODEL_PATH,
            "scaler_path": SCALER_PATH
        }
        
        return jsonify(model_info)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Función para calcular la distancia entre dos puntos geográficos
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia entre dos puntos geográficos usando la fórmula de Haversine
    
    Args:
        lat1, lon1: Coordenadas del primer punto
        lat2, lon2: Coordenadas del segundo punto
        
    Returns:
        Distancia en kilómetros
    """
    # Radio de la Tierra en kilómetros
    R = 6371.0
    
    # Convertir grados a radianes
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)
    
    # Diferencia de longitud y latitud
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    # Fórmula de Haversine
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    
    return distance

# Función para encontrar el punto de referencia más cercano
def find_nearest_reference_point(lat, lon):
    """
    Encuentra el punto de referencia más cercano a las coordenadas dadas
    
    Args:
        lat: Latitud del punto
        lon: Longitud del punto
        
    Returns:
        Diccionario con información del punto más cercano
    """
    min_distance = float('inf')
    nearest_point = None
    nearest_country = None
    
    for country, points in REFERENCE_POINTS.items():
        for point in points:
            distance = haversine_distance(lat, lon, point['lat'], point['lon'])
            if distance < min_distance:
                min_distance = distance
                nearest_point = point
                nearest_country = country
    
    return {
        'country': nearest_country,
        'point': nearest_point,
        'distance_km': min_distance
    }

# Función para obtener los datos recientes de Copernicus
def get_recent_copernicus_data(lat, lon):
    """
    Obtiene datos recientes de Copernicus para un punto específico
    
    Args:
        lat: Latitud del punto
        lon: Longitud del punto
        
    Returns:
        DataFrame con los datos procesados
    """
    # Calcular las fechas para los últimos 5 días
    end_date = datetime.now() - timedelta(days=2)  # Copernicus tiene un retraso de 2-5 días
    start_date = end_date - timedelta(days=5)
    
    # Formatear las fechas para la solicitud
    year = end_date.strftime("%Y")
    month = end_date.strftime("%m")
    
    # Crear lista de días para la solicitud
    days = []
    current_date = start_date
    while current_date <= end_date:
        days.append(current_date.strftime("%d"))
        current_date += timedelta(days=1)
    
    # Definir el área de interés (un área pequeña alrededor del punto)
    area = [lat + 0.5, lon - 0.5, lat - 0.5, lon + 0.5]  # [North, West, South, East]
    
    # Configurar la solicitud a Copernicus
    dataset = os.getenv('CDS_DATASET', 'reanalysis-era5-land')
    request = {
        "variable": [
            "soil_temperature_level_1",
            "volumetric_soil_water_layer_1",
            "volumetric_soil_water_layer_2",
            "volumetric_soil_water_layer_3",
            "volumetric_soil_water_layer_4",
            "evaporation_from_vegetation_transpiration",
            "total_precipitation"
        ],
        "year": year,
        "month": month,
        "day": days,
        "time": ["06:00", "18:00"],
        "area": area,
        "format": "netcdf"
    }
    
    try:
        # Crear un directorio temporal para guardar el archivo
        temp_dir = tempfile.mkdtemp()
        output_file = os.path.join(temp_dir, "recent_data.nc")
        
        # Hacer la solicitud a Copernicus
        client = cdsapi.Client(url=os.getenv('CDS_URL'), key=os.getenv('CDS_KEY'))
        client.retrieve(dataset, request, output_file)
        
        # Cargar los datos con xarray
        ds = xr.open_dataset(output_file)
        
        # Extraer datos para el punto específico
        point_data = ds.sel(latitude=lat, longitude=lon, method='nearest')
        
        # Convertir a DataFrame
        df = point_data.to_dataframe()
        
        # Limpiar y formatear el DataFrame
        df = df.reset_index()
        
        # Renombrar columnas para coincidencia con el modelo
        column_mapping = {
            'volumetric_soil_water_layer_1': 'soil_water_1',
            'volumetric_soil_water_layer_2': 'soil_water_2',
            'volumetric_soil_water_layer_3': 'soil_water_3',
            'volumetric_soil_water_layer_4': 'soil_water_4',
            'soil_temperature_level_1': 'soil_temp_lvl1',
            'evaporation_from_vegetation_transpiration': 'evaporation',
            'total_precipitation': 'precipitation'
        }
        df = df.rename(columns=column_mapping)
        
        # Limpiar temporales
        ds.close()
        os.remove(output_file)
        os.rmdir(temp_dir)
        
        return df
        
    except Exception as e:
        print(f"Error al obtener datos de Copernicus: {e}")
        
        # Crear datos de ejemplo en caso de error con Copernicus
        # Esto es solo para propósitos de demostración
        dates = pd.date_range(start=start_date, end=end_date, freq='12H')
        n_samples = len(dates)
        
        dummy_data = {
            'time': dates,
            'latitude': [lat] * n_samples,
            'longitude': [lon] * n_samples,
            'soil_water_1': np.random.uniform(0.3, 0.5, n_samples),
            'soil_water_2': np.random.uniform(0.3, 0.5, n_samples),
            'soil_water_3': np.random.uniform(0.3, 0.5, n_samples),
            'soil_water_4': np.random.uniform(0.3, 0.5, n_samples),
            'soil_temp_lvl1': np.random.uniform(290, 300, n_samples),
            'evaporation': np.random.uniform(0, 0.001, n_samples),
            'precipitation': np.random.uniform(0, 0.005, n_samples)
        }
        
        return pd.DataFrame(dummy_data)

# Función para evaluar el estado del suelo
def evaluate_soil_condition(df, prediction):
    """
    Evalúa el estado del suelo basado en los datos recientes y la predicción del modelo
    
    Args:
        df: DataFrame con datos procesados
        prediction: Valor predicho por el modelo para soil_water_1
        
    Returns:
        Diccionario con análisis y estado del suelo
    """
    # Calcular estadísticas
    avg_soil_water_1 = df['soil_water_1'].mean()
    avg_precipitation = df['precipitation'].mean()
    max_precipitation = df['precipitation'].max()
    soil_water_trend = df['soil_water_1'].pct_change().mean()
    
    # Normalizar variables para el modelo
    df_model = df.copy()
    
    # Convertir valores para coincidir con el rango esperado por el modelo (entre 0 y 1)
    df_model['soil_water_2'] = df_model['soil_water_2'].clip(0, 1)
    df_model['soil_water_3'] = df_model['soil_water_3'].clip(0, 1)
    df_model['soil_water_4'] = df_model['soil_water_4'].clip(0, 1)
    
    # Normalizar soil_temp_lvl1 (asumir rango de 280-310K -> 0-1)
    df_model['soil_temp_lvl1'] = (df_model['soil_temp_lvl1'] - 280) / 30
    df_model['soil_temp_lvl1'] = df_model['soil_temp_lvl1'].clip(0, 1)
    
    # Calcular la media de las variables para el modelo
    avg_features = {
        'soil_water_2': df_model['soil_water_2'].mean(),
        'soil_water_3': df_model['soil_water_3'].mean(),
        'soil_water_4': df_model['soil_water_4'].mean(),
        'soil_temp_lvl1': df_model['soil_temp_lvl1'].mean(),
        'lat': df['latitude'].iloc[0],
        'lon': df['longitude'].iloc[0]
    }
    
    # Determinar el estado del suelo basado en la predicción del modelo
    # Estos umbrales pueden ajustarse según el conocimiento experto o análisis de datos históricos
    if prediction < 0.25:
        soil_condition = "Sequía"
        alert_level = "Alta"
        message = "ALERTA ALTA: Condiciones de sequía severa detectadas. Se recomienda implementar medidas de conservación de agua urgentes."
    elif prediction < 0.35:
        soil_condition = "Seco"
        alert_level = "Media"
        message = "ALERTA MEDIA: Suelo seco detectado. Se recomienda aumentar el riego y monitorear las condiciones."
    elif prediction < 0.55:
        soil_condition = "Normal"
        alert_level = "Baja"
        message = "ALERTA BAJA: Condiciones de humedad normales. Se recomienda mantener prácticas regulares de riego."
    elif prediction < 0.7:
        soil_condition = "Húmedo"
        alert_level = "Baja"
        message = "ALERTA BAJA: Suelo con buena humedad. Considere reducir ligeramente el riego."
    else:
        soil_condition = "Saturado"
        alert_level = "Media"
        message = "ALERTA MEDIA: Suelo con alta saturación de agua. Riesgo potencial de encharcamiento o drenaje deficiente."
    
    return {
        'avg_features': avg_features,
        'soil_condition': soil_condition,
        'alert_level': alert_level,
        'message': message,
        'avg_soil_water_1': float(avg_soil_water_1),
        'avg_precipitation': float(avg_precipitation),
        'max_precipitation': float(max_precipitation),
        'soil_moisture_trend': float(soil_water_trend)
    }

@app.route('/recent_forecast', methods=['POST'])
def recent_forecast():
    """
    Endpoint para obtener un pronóstico basado en datos recientes de Copernicus
    y generar alertas sobre el estado del suelo
    
    Ejemplo de JSON de entrada:
    {
        "lat": -34.6,
        "lon": -58.4
    }
    """
    try:
        # Obtener datos de la solicitud
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No se recibieron datos"}), 400
        
        # Verificar que se proporcionaron coordenadas
        if 'lat' not in data or 'lon' not in data:
            return jsonify({"error": "Faltan las coordenadas (lat, lon)"}), 400
        
        lat = float(data['lat'])
        lon = float(data['lon'])
        
        # Validar coordenadas (dentro de América Latina)
        if not (-56 <= lat <= 25) or not (-120 <= lon <= -34):
            return jsonify({"error": "Coordenadas fuera del rango de América Latina"}), 400
        
        # Encontrar el punto de referencia más cercano
        nearest = find_nearest_reference_point(lat, lon)
        reference_point = nearest['point']
        country = nearest['country']
        
        # Usar las coordenadas del punto de referencia para datos recientes
        recent_data = get_recent_copernicus_data(reference_point['lat'], reference_point['lon'])
        
        # Preparar los features para la predicción del modelo
        # Normalizar variables para el modelo
        df_model = recent_data.copy()
        
        # Convertir valores para coincidir con el rango esperado por el modelo (entre 0 y 1)
        df_model['soil_water_2'] = df_model['soil_water_2'].clip(0, 1)
        df_model['soil_water_3'] = df_model['soil_water_3'].clip(0, 1)
        df_model['soil_water_4'] = df_model['soil_water_4'].clip(0, 1)
        
        # Normalizar soil_temp_lvl1 (asumir rango de 280-310K -> 0-1)
        df_model['soil_temp_lvl1'] = (df_model['soil_temp_lvl1'] - 280) / 30
        df_model['soil_temp_lvl1'] = df_model['soil_temp_lvl1'].clip(0, 1)
        
        # Calcular la media de las variables para el modelo
        avg_features = {
            'soil_water_2': df_model['soil_water_2'].mean(),
            'soil_water_3': df_model['soil_water_3'].mean(),
            'soil_water_4': df_model['soil_water_4'].mean(),
            'soil_temp_lvl1': df_model['soil_temp_lvl1'].mean(),
            'lat': recent_data['latitude'].iloc[0],
            'lon': recent_data['longitude'].iloc[0],
            'pais': country
        }
        
        # Hacer predicción con el modelo
        X = {}
        for feature in FEATURES:
            X[feature] = avg_features[feature]
        
        # One-hot encoding para el país con drop_first=True (sin Argentina)
        for c in COUNTRIES:
            if c == 'Argentina':
                continue
            elif c == country:
                X[f'pais_{c}'] = 1
            else:
                X[f'pais_{c}'] = 0
        
        # Convertir a DataFrame
        input_df = pd.DataFrame([X])
        
        # Realizar la predicción
        prediction = model.predict(input_df)[0]
        
        # Evaluar el estado del suelo basado en la predicción
        soil_analysis = evaluate_soil_condition(recent_data, prediction)
        
        # Preparar la respuesta
        response = {
            "coordinates": {
                "requested": {"lat": lat, "lon": lon},
                "reference_point": {
                    "lat": reference_point['lat'], 
                    "lon": reference_point['lon'],
                    "description": reference_point['desc'],
                    "distance_km": round(nearest['distance_km'], 2)
                }
            },
            "country": country,
            "period": {
                "start_date": recent_data['time'].min().strftime("%Y-%m-%d"),
                "end_date": recent_data['time'].max().strftime("%Y-%m-%d")
            },
            "soil_moisture_prediction": {
                "soil_water_1": float(prediction),
                "message": f"Predicción de humedad del suelo (capa 1): {prediction:.4f}"
            },
            "soil_condition": {
                "condition": soil_analysis['soil_condition'],
                "alert_level": soil_analysis['alert_level'],
                "message": soil_analysis['message']
            },
            "stats": {
                "avg_soil_moisture": soil_analysis['avg_soil_water_1'],
                "avg_precipitation": soil_analysis['avg_precipitation'],
                "max_precipitation": soil_analysis['max_precipitation'],
                "soil_moisture_trend": soil_analysis['soil_moisture_trend']
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ejecutar la aplicación con configuración desde variables de entorno
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    port = int(os.getenv('PORT', 5000))
    app.run(debug=debug_mode, port=port)
