import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

print("Iniciando o laboratório de Machine Learning para ABDOMINAIS...")

# 1. Carregamento da Matriz de Dados
try:
    df = pd.read_csv('dataset_abdominal.csv')
    print(f"Dataset carregado com sucesso. Total de frames anotados: {len(df)}")
except FileNotFoundError:
    print("ERRO CRÍTICO: O ficheiro 'dataset_abdominal.csv' não foi encontrado.")
    exit()

# 2. Limpeza Estrutural
df = df.dropna()

# 3. Separação Espacial: X (Características) e Y (Alvo)
colunas_features = [
    'ang_esq', 'ang_dir', 
    'ombro_esq_x', 'ombro_esq_y', 'quadril_esq_x', 'quadril_esq_y', 'joelho_esq_x', 'joelho_esq_y',
    'ombro_dir_x', 'ombro_dir_y', 'quadril_dir_x', 'quadril_dir_y', 'joelho_dir_x', 'joelho_dir_y'
]

X = df[colunas_features]
Y = df['label']

# 4. Validação Cruzada (80% para estudo, 20% para a prova de fogo)
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=42)

# 5. O Treinamento Matemático (Calculando o novo Hiperplano)
print("Treinando o modelo Support Vector Machine (SVM) focado no Tronco...")
modelo = SVC(kernel='linear', random_state=42)
modelo.fit(X_treino, Y_treino)

# 6. A Prova Final (Auditoria)
previsoes = modelo.predict(X_teste)
acuracia = accuracy_score(Y_teste, previsoes)

print("-" * 50)
print(f"ESTATÍSTICA: Acurácia do Modelo = {acuracia * 100:.2f}%")
print("-" * 50)

# 7. O Deploy do Cérebro
with open('modelo_svm_abdominal.pkl', 'wb') as arquivo:
    pickle.dump(modelo, arquivo)

print("\nSegundo Cérebro exportado com sucesso! O ficheiro 'modelo_svm_abdominal.pkl' nasceu na raiz do projeto.")