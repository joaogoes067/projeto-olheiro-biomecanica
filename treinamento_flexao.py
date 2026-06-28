import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

print("Iniciando o laboratório de Machine Learning...")

# 1. Carrega os dados que você anotou
df = pd.read_csv('dataset_flexao.csv')
print(f"Dataset carregado com sucesso. Total de frames anotados: {len(df)}")
df = df.dropna()

# 2. Separa a Geometria (X) do Rótulo (Y)
colunas_features = [
    'ang_esq', 'ang_dir', 
    'ombro_esq_x', 'ombro_esq_y', 'cotovelo_esq_x', 'cotovelo_esq_y', 'pulso_esq_x', 'pulso_esq_y',
    'ombro_dir_x', 'ombro_dir_y', 'cotovelo_dir_x', 'cotovelo_dir_y', 'pulso_dir_x', 'pulso_dir_y'
]

X = df[colunas_features]
Y = df['label']

# 3. Divide os dados para treino e prova
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=42)

# 4. Treina o Algoritmo SVM
print("Treinando o modelo Support Vector Machine (SVM)...")
modelo = SVC(kernel='linear', random_state=42)
modelo.fit(X_treino, Y_treino)

# 5. Aplica a Prova Final
previsoes = modelo.predict(X_teste)
acuracia = accuracy_score(Y_teste, previsoes)

print("-" * 50)
print(f"RESULTADO: Acurácia do Modelo = {acuracia * 100:.2f}%")
print("-" * 50)

# 6. Salva o Cérebro treinado
with open('modelo_svm.pkl', 'wb') as arquivo:
    pickle.dump(modelo, arquivo)

print("\nCérebro exportado com sucesso! O arquivo 'modelo_svm.pkl' foi criado.")