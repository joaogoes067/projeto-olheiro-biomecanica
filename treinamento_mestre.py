import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC # <-- MOTOR OTIMIZADO PARA GRANDES DADOS
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings

warnings.filterwarnings("ignore")

print("Iniciando o laboratório estatístico do CLASSIFICADOR MESTRE (Olheiro)...")

try:
    df = pd.read_csv('dataset_olheiro.csv')
    print(f"Super-Dataset carregado com sucesso. Total de frames universais: {len(df)}")
except FileNotFoundError:
    print("ERRO CRÍTICO: O ficheiro 'dataset_olheiro.csv' não encontrado.")
    exit()

df = df.dropna()

colunas_features = [
    "ombro_esq_x", "ombro_esq_y", "ombro_dir_x", "ombro_dir_y",
    "cotovelo_esq_x", "cotovelo_esq_y", "cotovelo_dir_x", "cotovelo_dir_y",
    "pulso_esq_x", "pulso_esq_y", "pulso_dir_x", "pulso_dir_y",
    "quadril_esq_x", "quadril_esq_y", "quadril_dir_x", "quadril_dir_y",
    "joelho_esq_x", "joelho_esq_y", "joelho_dir_x", "joelho_dir_y",
    "tornozelo_esq_x", "tornozelo_esq_y", "tornozelo_dir_x", "tornozelo_dir_y"
]

X = df[colunas_features]
Y = df['label_exercicio']

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Calculando hiperplanos multiclasse no espaço R^24 (Motor Otimizado LinearSVC)...")
# O parametro dual=False é o segredo matemático para acelerar quando temos mais amostras do que dimensões
modelo_olheiro = LinearSVC(random_state=42, max_iter=10000, dual=False)
modelo_olheiro.fit(X_treino, Y_treino)

previsoes = modelo_olheiro.predict(X_teste)
acuracia = accuracy_score(Y_teste, previsoes)

print("-" * 60)
print(f"CONVERGÊNCIA DO OLHEIRO: Acurácia Global do Sistema = {acuracia * 100:.2f}%")
print("-" * 60)
print("\nRelatório de Precisão por Tipo de Exercício:")
print("1: Flexão | 2: Abdominal | 3: Agachamento")
print(classification_report(Y_teste, previsoes))
print("-" * 60)

with open('modelo_svm_olheiro.pkl', 'wb') as arquivo:
    pickle.dump(modelo_olheiro, arquivo)

print("\nCérebro Mestre exportado com sucesso! O ficheiro 'modelo_svm_olheiro.pkl' nasceu na raiz do projeto.")