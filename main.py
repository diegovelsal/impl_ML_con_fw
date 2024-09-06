import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb

# Función para modificar los datos para el entrenamiento del modelo
def data_engineering(df):
    # Eliminación de columnas no necesarias
    df = df.drop(['Unnamed: 0', 'comp', 'round', 'attendance', 'match report', 'notes', 'time'], axis=1)

    # Columnas que pudieran ser relevantes pero por el momento no se utilizarán
    df = df.drop(['captain', 'formation', 'referee'], axis=1)

    # Cambio de tipo de datos para las columnas
    df['date'] = pd.to_datetime(df['date'])
    df['venue'] = df['venue'].astype('category')
    df['opponent'] = df['opponent'].astype('category')
    df['team'] = df['team'].astype('category')
    df['result'] = df['result'].astype('category')

    # Corregir variable 'day'
    df['day'] = df['date'].dt.day_name()
    df['day'] = df['day'].astype('category')

    # Corregir variable 'season'
    df.drop_duplicates(subset=df.columns.drop('season'), inplace=True)
    df['season'] = pd.to_datetime(df['date']).apply(lambda date: date.year + 1 if date.month >= 8 else date.year)

    # Eliminar columnas de date que ya no se utilizará
    df = df.drop(['date'], axis=1)

    # Eliminar donde la variable 'dist' sea nulos
    # df = df.dropna(subset=['dist'])

    # Para la variable 'Result', cambiarla a 1 si es victoria y a 0 si es empate o derrota
    df['result'] = df['result'].apply(lambda result: 1 if result == 'W' else 0)
    df['result'] = df['result'].astype('bool')

    # Convertir variables categóricas a numéricas con get_dummies
    df = pd.get_dummies(df, drop_first=False)
    df = df.drop(['venue_Away'], axis=1)

    print(df.info())

    return df

# Función para separar los datos en entrenamiento, validación y prueba
def split_data(df):
    # Separar las variables independientes y dependientes
    X = df.drop(['result', 'gf', 'ga'], axis=1)
    y = df['result']

    # Separar los datos en entrenamiento (60%), validación (20%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.16, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Función para escalar los datos
def scale_data(X_train, X_val, X_test):
    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled

# Función para entrenar y evaluar el modelo usando XGBoost
def train_and_evaluate_model(X_train_scaled, X_val_scaled, y_train, y_val, X_test_scaled, y_test):
    # Crear DMatrix para XGBoost
    train_dmatrix = xgb.DMatrix(X_train_scaled, label=y_train)
    val_dmatrix = xgb.DMatrix(X_val_scaled, label=y_val)
    test_dmatrix = xgb.DMatrix(X_test_scaled, label=y_test)
    
    # Parámetros de XGBoost con regularización
    params = {
        'objective': 'binary:logistic',  # Clasificación binaria
        'eval_metric': 'logloss',
        'max_depth': 6,                  # Profundidad máxima de los árboles
        'learning_rate': 0.01,            # Tasa de aprendizaje
        'lambda': 1,                     # Regularización L2
        'alpha': 0.5  
    }

    # Entrenar el modelo
    model = xgb.train(params, train_dmatrix, num_boost_round=100, evals=[(val_dmatrix, 'eval')])

    # Predecir en el conjunto de validación
    y_val_pred_prob = model.predict(val_dmatrix)
    y_val_pred = (y_val_pred_prob > 0.5).astype(int)
    
    # Evaluar en el conjunto de validación
    accuracy_val = accuracy_score(y_val, y_val_pred)
    confusion_val = confusion_matrix(y_val, y_val_pred)
    report_val = classification_report(y_val, y_val_pred, output_dict=True)

    print('Validation Accuracy:', accuracy_val)
    print('Validation Confusion Matrix:\n', confusion_val)
    print('Validation Classification Report:\n', classification_report(y_val, y_val_pred))

    # Predecir en el conjunto de prueba
    y_test_pred_prob = model.predict(test_dmatrix)
    y_test_pred = (y_test_pred_prob > 0.5).astype(int)

    # Evaluar en el conjunto de prueba
    accuracy_test = accuracy_score(y_test, y_test_pred)
    confusion_test = confusion_matrix(y_test, y_test_pred)
    report_test = classification_report(y_test, y_test_pred, output_dict=True)

    print('Test Accuracy:', accuracy_test)
    print('Test Confusion Matrix:\n', confusion_test)
    print('Test Classification Report:\n', classification_report(y_test, y_test_pred))

    return accuracy_val, confusion_val, report_val, accuracy_test, confusion_test, report_test

# Función para plotear las métricas
def plot_metrics():
    # Graficar la matriz de confusión de validación
    plt.figure(figsize=(12, 5))

def main():
    # Carga de datos
    df = pd.read_csv('data/matches.csv')

    # Data Engineering
    df = data_engineering(df)

    # Almacenar datos en un archivo CSV
    # df.to_csv('data/matches_clean.csv', index=False)

    # Separar y escalar los datos
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_scaled, X_val_scaled, X_test_scaled = scale_data(X_train, X_val, X_test)
    
    # Plotear las métricas
    plot_metrics()

if __name__ == '__main__':
    main()