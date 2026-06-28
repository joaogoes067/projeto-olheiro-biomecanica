import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import warnings

warnings.filterwarnings("ignore")

print("Iniciando o laboratório de Machine Learning para AGACHAMENTOS...")

# 1. Carregamento da Matriz de Dados
try:
    df = pd.read_csv('dataset_agachamento.csv')
    print(f"Dataset carregado com sucesso. Total de frames anotados: {len(df)}")
except FileNotFoundError:
    print("ERRO CRÍTICO: O ficheiro 'dataset_agachamento.csv' não foi encontrado.")
    exit()

# 2. Limpeza Estrutural
df = df.dropna()

# 3. Separação Espacial: X (Geometria Inferior) e Y (Rótulo do Atleta)
colunas_features = [
    'ang_esq', 'ang_dir', 
    'quadril_esq_x', 'quadril_esq_y', 'joelho_esq_x', 'joelho_esq_y', 'tornozelo_esq_x', 'tornozelo_esq_y',
    'quadril_dir_x', 'quadril_dir_y', 'joelho_dir_x', 'joelho_dir_y', 'tornozelo_dir_x', 'tornozelo_dir_y'
]

X = df[colunas_features]
Y = df['label']

# 4. Auditoria de Dados (80% Estudo, 20% Prova)
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=42)

# 5. O Treinamento (Otimização Convexa do Hiperplano)
print("Treinando o modelo Support Vector Machine (SVM) focado nos Membros Inferiores...")
modelo = SVC(kernel='linear', random_state=42)
modelo.fit(X_treino, Y_treino)

# 6. O Veredito
previsoes = modelo.predict(X_teste)
acuracia = accuracy_score(Y_teste, previsoes)

print("-" * 50)
print(f"VITÓRIA ESTATÍSTICA: Acurácia do Modelo = {acuracia * 100:.2f}%")
print("-" * 50)

# 7. O Deploy
with open('modelo_svm_agachamento.pkl', 'wb') as arquivo:
    pickle.dump(modelo, arquivo)

print("\nTerceiro Cérebro exportado com sucesso! O ficheiro 'modelo_svm_agachamento.pkl' nasceu na raiz.")