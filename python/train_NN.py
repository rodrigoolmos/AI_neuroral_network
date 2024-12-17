import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model


# 1. Cargar el dataset
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    # Separar características (X) y etiqueta (y) - asumiendo que la última columna es la etiqueta
    X = df.iloc[:, :-1].values  # Todas las columnas excepto la última
    y = df.iloc[:, -1].values   # Última columna como etiquetas
    return X, y

# 2. Preprocesar los datos
def preprocess_data(X, y):
    # Normalizar características
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Asegurar que las etiquetas sean binarias (0 y 1)
    y = np.where(y > 0, 1, 0)
    return X, y

# 3. Crear la red neuronal
def create_nn(input_dim):
    model = Sequential()
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform', input_dim=input_dim))    # Capa oculta 1
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform',))                         # Capa oculta 2
    model.add(Dense(50, activation='relu', kernel_initializer='he_uniform',))                          # Capa oculta 3

    model.add(Dense(1, activation='sigmoid'))                       # Capa de salida (clasificación binaria)
    
    # Compilar el modelo
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# 4. Entrenar y evaluar la red neuronal
def train_and_evaluate(csv_path):
    # Cargar datos
    X, y = load_dataset(csv_path)
    
    # Preprocesar datos
    X, y = preprocess_data(X, y)
    
    # Dividir en entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear la red neuronal
    model = create_nn(X_train.shape[1])
    
    # Entrenar el modelo
    print("Entrenando la red neuronal...")
    model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=1)

    # Evaluar el modelo
    print("\nEvaluando el modelo en el conjunto de prueba...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Pérdida en prueba: {loss:.4f}")
    print(f"Precisión en prueba: {accuracy:.4f}")

    return model

# 5. Ruta del archivo CSV y ejecución
if __name__ == "__main__":
    csv_path = "/home/rodrigo/Documents/AI_neuroral_network/datasets/kaggle/alzheimers_processed_dataset.csv"  # Cambia esto por la ruta de tu archivo CSV
    model = train_and_evaluate(csv_path)
