import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_engineering(df):
    # Drop colums street, country, date and statezip
    df.drop(['street', 'country', 'date', 'statezip'], axis=1, inplace=True)

    # yr_renovated
    df['yr_renovated'] = df.apply(lambda x: x['yr_built'] if x['yr_renovated'] < x['yr_built'] else x['yr_renovated'], axis=1)
    df['yr_renovated'] = df.apply(lambda x: x['yr_built'] if x['yr_renovated'] == 0 else x['yr_renovated'], axis=1)

    # Basement
    bins = [0, 1, 500, 1000, 1500, 2000, float('inf')]
    labels = [0, 1, 2, 3, 4, 5]
    df['basement'] = pd.cut(df['sqft_basement'], bins=bins, labels=labels, right=False)
    
    # Nueva columna Seattle que es 1 si la ciudad es Seattle y 0 en otro caso
    df['Seattle'] = df['city'].apply(lambda x: 1 if x == 'Seattle' else 0)
    df.drop('city', axis=1, inplace=True)

    return df

def main():
    # Carga de datos
    # df = pd.read_csv('data/usa_housing.csv')

    # Data Engineering para la preparación del modelo
    # df = data_engineering(df)

    # Save the data to a new csv file
    # df.to_csv('data/usa_housing_engineered.csv', index=False)

    # Carga de datos
    df = pd.read_csv('data/usa_housing_engineered.csv')

    # Definición de X y Y
    X = df.drop('price', axis=1)
    y = df['price']

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Datos preparados para el modelado")

if __name__ == '__main__':
    main()

'''
['Seattle', 'Carnation', 'Issaquah', 'Maple Valley', 'Kent', 'Redmond', 'Clyde Hill', 'Shoreline', 'Mercer Island', 'Auburn', 'Bellevue', 'Duvall', 'Renton', 'Sammamish', 'Woodinville', 'Kirkland', 'Burien', 'Federal Way', 'Normandy Park', 'Vashon', 'Kenmore', 'Yarrow Point', 'SeaTac', 'Newcastle', 'Covington', 'Lake Forest Park', 'Snoqualmie', 'Des Moines', 'Fall City', 'Ravensdale', 'Tukwila', 'North Bend', 'Medina', 'Bothell', 'Enumclaw','Snoqualmie Pass', 'Pacific', 'Black Diamond', 'Beaux Arts Village', 'Algona','Preston', 'Milton', 'Skykomish']
'''
