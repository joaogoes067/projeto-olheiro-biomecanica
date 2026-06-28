import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector
import pickle
import warnings

warnings.filterwarnings("ignore")

print("Iniciando o Ecossistema Biomecânico Unificado (Nível 1 + Nível 2)...")

# 1. CARREGAMENTO DA NOSSA BIBLIOTECA COMPLETA
with open('modelo_svm_olheiro.pkl', 'rb') as f: modelo_olheiro = pickle.load(f)
with open('modelo_svm.pkl', 'rb') as f: cerebro_flexao = pickle.load(f) # Cérebro das flexões
with open('modelo_svm_abdominal.pkl', 'rb') as f: cerebro_abdominal = pickle.load(f)
with open('modelo_svm_agachamento.pkl', 'rb') as f: cerebro_agachamento = pickle.load(f)

detector = PoseDetector(trackCon=0.70, detectionCon=0.70)

# Puxando a prova gravada (O Vídeo Supremo da Banca)
cap = cv2.VideoCapture('apresentacao_final3.mp4')

# 2. CONTADORES E MÁQUINAS DE ESTADO INDEPENDENTES
contadores = {1: 0.0, 2: 0.0, 3: 0.0} # 1=Flexão, 2=Abdominal, 3=Agachamento
estados = {1: 0, 2: 0, 3: 0}          # 0 = Repouso, 1 = Contração

classes_nomes = {
    1: ("FLEXAO DE BRACO", (255, 0, 0)),
    2: ("ABDOMINAL", (0, 165, 255)),
    3: ("AGACHAMENTO", (0, 255, 0))
}

# 3. FUNÇÃO MATEMÁTICA UNIVERSAL
def calcular_angulo_2d(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0
    cosseno_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
    return np.degrees(np.arccos(np.clip(cosseno_theta, -1.0, 1.0)))

while True:
    ret, img = cap.read()
    if not ret: 
        # Quando o vídeo acabar, ele reinicia automaticamente para a banca continuar avaliando
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        contadores = {1: 0.0, 2: 0.0, 3: 0.0}
        estados = {1: 0, 2: 0, 3: 0}
        continue

    # Blindagem de Escala
    h, w = img.shape[:2]
    nova_altura = 700
    proporcao = nova_altura / h
    nova_largura = int(w * proporcao)
    img = cv2.resize(img, (nova_largura, nova_altura))

    img = detector.findPose(img, draw=True)
    lmList, bbox = detector.findPosition(img, draw=False, bboxWithHands=False)

    if len(lmList) != 0:
        # =================================================================
        # CAMADA NÍVEL 1: O OLHEIRO AVALIA O CORPO INTEIRO (R^24)
        # =================================================================
        pontos_olheiro = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        features_olheiro = []
        for p in pontos_olheiro:
            features_olheiro.extend(lmList[p][1:3])
        
        classe_atual = modelo_olheiro.predict([features_olheiro])[0]
        nome_exercicio, cor_exercicio = classes_nomes.get(classe_atual, ("DESCONHECIDO", (0,0,255)))

        # =================================================================
        # CAMADA NÍVEL 2: ROTEAMENTO E CONTAGEM ISOLADA (R^14)
        # =================================================================
        
        # --- SE FOR FLEXÃO ---
        if classe_atual == 1:
            o_esq, c_esq, p_esq = lmList[11][1:3], lmList[13][1:3], lmList[15][1:3]
            o_dir, c_dir, p_dir = lmList[12][1:3], lmList[14][1:3], lmList[16][1:3]
            
            ang_esq = calcular_angulo_2d(o_esq, c_esq, p_esq)
            ang_dir = calcular_angulo_2d(o_dir, c_dir, p_dir)
            
            feat_flexao = [ang_esq, ang_dir, o_esq[0], o_esq[1], c_esq[0], c_esq[1], p_esq[0], p_esq[1],
                           o_dir[0], o_dir[1], c_dir[0], c_dir[1], p_dir[0], p_dir[1]]
            
            predicao = cerebro_flexao.predict([feat_flexao])[0]
            
            if predicao == 1 and estados[1] == 0:
                contadores[1] += 0.5; estados[1] = 1
            elif predicao == 0 and estados[1] == 1:
                contadores[1] += 0.5; estados[1] = 0

        # --- SE FOR ABDOMINAL ---
        elif classe_atual == 2:
            o_esq, q_esq, j_esq = lmList[11][1:3], lmList[23][1:3], lmList[25][1:3]
            o_dir, q_dir, j_dir = lmList[12][1:3], lmList[24][1:3], lmList[26][1:3]
            
            ang_esq = calcular_angulo_2d(o_esq, q_esq, j_esq)
            ang_dir = calcular_angulo_2d(o_dir, q_dir, j_dir)
            
            feat_abd = [ang_esq, ang_dir, o_esq[0], o_esq[1], q_esq[0], q_esq[1], j_esq[0], j_esq[1],
                        o_dir[0], o_dir[1], q_dir[0], q_dir[1], j_dir[0], j_dir[1]]
            
            predicao = cerebro_abdominal.predict([feat_abd])[0]
            
            if predicao == 1 and estados[2] == 0:
                contadores[2] += 0.5; estados[2] = 1
            elif predicao == 0 and estados[2] == 1:
                contadores[2] += 0.5; estados[2] = 0

        # --- SE FOR AGACHAMENTO ---
        elif classe_atual == 3:
            q_esq, j_esq, t_esq = lmList[23][1:3], lmList[25][1:3], lmList[27][1:3]
            q_dir, j_dir, t_dir = lmList[24][1:3], lmList[26][1:3], lmList[28][1:3]
            
            ang_esq = calcular_angulo_2d(q_esq, j_esq, t_esq)
            ang_dir = calcular_angulo_2d(q_dir, j_dir, t_dir)
            
            feat_agacha = [ang_esq, ang_dir, q_esq[0], q_esq[1], j_esq[0], j_esq[1], t_esq[0], t_esq[1],
                           q_dir[0], q_dir[1], j_dir[0], j_dir[1], t_dir[0], t_dir[1]]
            
            predicao = cerebro_agachamento.predict([feat_agacha])[0]
            
            if predicao == 1 and estados[3] == 0:
                contadores[3] += 0.5; estados[3] = 1
            elif predicao == 0 and estados[3] == 1:
                contadores[3] += 0.5; estados[3] = 0

        # =================================================================
        # INTERFACE GRÁFICA GERAL (BLINDADA PARA VÍDEOS VERTICAIS)
        # =================================================================
        cv2.rectangle(img, (0, 0), (nova_largura, 130), (0, 0, 0), cv2.FILLED)
        
        cv2.putText(img, f"POSTURA: {nome_exercicio}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, cor_exercicio, 2)
        
        # Contadores empilhados na mesma coluna (x=10) para evitar o esmagamento lateral
        cv2.putText(img, f"Flexoes: {int(contadores[1])}", (10, 65), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, f"Abdominais: {int(contadores[2])}", (10, 90), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, f"Agachamentos: {int(contadores[3])}", (10, 115), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Projeto Final - IA Biomecanica", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()