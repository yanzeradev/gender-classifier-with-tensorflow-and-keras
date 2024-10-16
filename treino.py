import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np
import os

# Definir o caminho para o diretório de imagens
imagens_dir = 'imagens'

# Inicializar listas para armazenar as imagens e os rótulos
imagens = []
rotulos = []

# Função para carregar as imagens e os rótulos
def carregar_imagens(diretorio):
    for subdir in os.listdir(diretorio):
        if subdir == 'men':
            rotulo = 0
        elif subdir == 'women':
            rotulo = 1
        else:
            continue
            
        for arquivo in os.listdir(os.path.join(diretorio, subdir)):
            img_path = os.path.join(diretorio, subdir, arquivo)
            img = Image.open(img_path)
            img = img.resize((224, 224))  # Redimensionar as imagens para 224x224
            img = np.array(img) / 255.0  # Normalizar as imagens
            imagens.append(img)
            rotulos.append(rotulo)

# Carregar as imagens de cada uma das pastas (train, valid, test)
for folder in ['train', 'valid', 'test']:
    carregar_imagens(os.path.join(imagens_dir, folder))

# Converter as listas para arrays do NumPy
imagens = np.array(imagens)
rotulos = np.array(rotulos)

# Dividir os dados em treinamento e teste
imagens_treinamento, imagens_teste, rotulos_treinamento, rotulos_teste = train_test_split(imagens, rotulos, test_size=0.2, random_state=42)

# Criar a rede neural
modelo = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compilar o modelo
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
modelo.fit(imagens_treinamento, rotulos_treinamento, epochs=12, batch_size=32, validation_data=(imagens_teste, rotulos_teste))

# Avaliar o modelo
previsoes = modelo.predict(imagens_teste)
classe = (previsoes > 0.5).astype(int)
acuracia = accuracy_score(rotulos_teste, classe)
print(f'Acurácia: {acuracia:.2f}')

# Salvar o modelo treinado
modelo.save('modelo_treinado.h5')
print("Modelo salvo como 'modelo_treinado.h5'")
