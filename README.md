# Projeto Olheiro: Análise Biomecânica via Inteligência Artificial

Este repositório documenta a implementação de um sistema de visão computacional e análise biomecânica utilizando máquinas de vetores de suporte (SVM) para a classificação e correção de movimentos corporais em tempo real.

## 🧠 Arquitetura do Sistema
O sistema foi projetado para decompor movimentos complexos em variáveis escalares, processando dados cinemáticos através de um classificador hierárquico:

* **Módulos de Classificação:**
    * **Cérebro 1 (Flexão):** Foco em 14 variáveis de membros superiores.
    * **Cérebro 2 (Abdominal):** Foco em 14 variáveis do tronco.
    * **Cérebro 3 (Agachamento):** Foco em 14 variáveis de membros inferiores.
    * **Olheiro (Classificador Principal):** Integração global com 24 variáveis para validação de alta precisão.

## 📊 Métricas de Performance
O modelo demonstra alta robustez com o processamento de mais de 100 mil quadros (frames) de vídeo:

| Modelo | Variáveis | Acurácia Global |
| :--- | :--- | :--- |
| Flexão | 14 | 80,87% |
| Abdominal | 14 | 79,82% |
| Agachamento | 14 | 88,17% |
| **Olheiro (Global)** | **24** | **96,01%** |

## 🛠 Tecnologias Utilizadas
* **Linguagem:** Python 3.x
* **Core de IA:** Scikit-learn (Support Vector Machines - LinearSVC)
* **Processamento:** Bibliotecas de visão computacional para extração de keypoints.
* **Documentação:** LaTeX (Relatórios técnicos e derivações matemáticas).

## 🚀 Como Executar
1. Clone este repositório:
   `git clone https://github.com/joaogoes067/projeto-olheiro-biomecanica.git`
2. Instale as dependências:
   `pip install -r requirements.txt`
3. Execute o script principal:
   `python deploy_unificado.py`

---
*Projeto desenvolvido como parte da pesquisa em Mestrado em Matemática.*