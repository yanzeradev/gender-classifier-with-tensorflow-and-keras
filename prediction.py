import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tkinter as tk
from tkinter import filedialog
import sys

# Carregar o modelo treinado
modelo = keras.models.load_model('modelo_treinado.h5')

# Função para carregar e pré-processar a imagem
def carregar_imagem(imagem_path):
    img = Image.open(imagem_path)
    img = img.resize((224, 224))  # Redimensionar para 224x224
    img = np.array(img) / 255.0  # Normalizar a imagem
    img = np.expand_dims(img, axis=0)  # Adicionar uma dimensão para o batch
    return img

# Função para adicionar texto à imagem
def adicionar_texto_imagem(imagem_path, texto):
    img = Image.open(imagem_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 48)
    texto_pos = (10, 10)
    
    # Adicionar texto
    draw.text(texto_pos, texto, fill="red", font=font)
    
    return img

# Função para abrir o seletor de arquivos e processar a imagem
def selecionar_imagem():
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True) 
    root.update()
    file_path = filedialog.askopenfilename() 
    if file_path:
        # Carregar e pré-processar a imagem
        imagem = carregar_imagem(file_path)
        
        # Fazer a predição
        previsao = modelo.predict(imagem)
        classe = (previsao > 0.5).astype(int)  # Convertendo a previsão para 0 ou 1

        # Determinar a classe
        if classe[0][0] == 0:
            label = "Predição: Homem"
        else:
            label = "Predição: Mulher"

        # Adicionar texto à imagem
        imagem_com_texto = adicionar_texto_imagem(file_path, label)

        # Mostrar a imagem com a predição
        imagem_com_texto.show()
    else:
        print("Nenhum arquivo foi selecionado.")

# Função para exibir o menu
def exibir_menu():
    while True:
        print("Menu:")
        print("1. Upar uma imagem")
        print("2. Sair")
        escolha = input("Escolha uma opção: ")

        if escolha == '1':
            selecionar_imagem()
        elif escolha == '2':
            print("Saindo...")
            sys.exit()
        else:
            print("Opção inválida. Tente novamente.")

# Executar o menu
exibir_menu()
