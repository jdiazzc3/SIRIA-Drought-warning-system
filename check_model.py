import pickle
import pandas as pd
import numpy as np

# Cargar el modelo
model_path = 'drought_mlp_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Intentar obtener los nombres de las características
if hasattr(model, 'feature_names_in_'):
    print("Nombres de características esperados por el modelo:")
    for name in model.feature_names_in_:
        print(f"- {name}")
else:
    print("El modelo no tiene el atributo feature_names_in_")

# Si se usó un pipeline o ColumnTransformer, podría estar ahí
try:
    if hasattr(model, 'named_steps'):
        for name, step in model.named_steps.items():
            if hasattr(step, 'feature_names_in_'):
                print(f"Características en el paso {name}:")
                for feat in step.feature_names_in_:
                    print(f"- {feat}")
except:
    pass

print("\nInformación sobre el modelo:")
print(f"Tipo: {type(model).__name__}")

# Prueba con un conjunto de datos mínimo
try:
    # Crear un DataFrame de ejemplo con las características básicas
    sample = pd.DataFrame({
        'soil_water_2': [0.4],
        'soil_water_3': [0.35],
        'soil_water_4': [0.3],
        'soil_temp_lvl1': [0.7],
        'lat': [-34.6],
        'lon': [-58.4]
    })
    
    # Agregar columnas dummy para todos los países excepto el primero (drop_first=True)
    countries = ['Argentina', 'Bolivia', 'Brasil', 'Chile', 'Colombia', 'Costa Rica', 'Cuba', 
                'Ecuador', 'El Salvador', 'Guatemala', 'Honduras', 'México', 'Nicaragua', 
                'Panamá', 'Paraguay', 'Perú', 'Puerto Rico', 'República Dominicana', 
                'Uruguay', 'Venezuela']
    
    # Versión 1: Agregar todos los países
    sample1 = sample.copy()
    for country in countries:
        sample1[f'pais_{country}'] = 0
    sample1['pais_Argentina'] = 1
    
    print("\nPrueba con todas las columnas de países:")
    try:
        model.predict(sample1)
        print("✓ Predicción exitosa")
        print(f"Columnas usadas: {sample1.columns.tolist()}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Versión 2: Agregar todos los países excepto el primero (drop_first=True)
    sample2 = sample.copy()
    for country in countries[1:]:  # Todos menos Argentina
        sample2[f'pais_{country}'] = 0
    
    print("\nPrueba con columnas de países (drop_first=True):")
    try:
        model.predict(sample2)
        print("✓ Predicción exitosa")
        print(f"Columnas usadas: {sample2.columns.tolist()}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
except Exception as e:
    print(f"Error durante la prueba: {e}")
