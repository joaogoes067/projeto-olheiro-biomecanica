import cv2
import numpy as np
import cvzone
from cvzone.PoseModule import PoseDetector
import pickle
import warnings

# Ignora os avisos de terminal estruturais do scikit-learn
warnings.filterwarnings("ignore")

class MotorInferencia:
    def __init__(self):
        self.detector = PoseDetector(trackCon=0.70, detectionCon=0.70)
        self.contador = 0
        self.direcao = 0  
        
        # Carregando o Cérebro Estatístico que você acabou de treinar
        print("Injetando modelo SVM na memória de vídeo...")
        with open('modelo_svm.pkl', 'rb') as f:
            self.modelo = pickle.load(f)

    @staticmethod
    def calcular_angulo_2d(p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: return 0
        cosseno_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        return np.degrees(np.arccos(np.clip(cosseno_theta, -1.0, 1.0)))

    def processar_frame(self, img):
        img = self.detector.findPose(img, draw=True)
        lmList, bbox = self.detector.findPosition(img, draw=False, bboxWithHands=False)

        if len(lmList) != 0:
            ombro_esq = lmList[11][1:3]
            cotovelo_esq = lmList[13][1:3]
            pulso_esq = lmList[15][1:3]
            
            ombro_dir = lmList[12][1:3]
            cotovelo_dir = lmList[14][1:3]
            pulso_dir = lmList[16][1:3]

            ang_esq = self.calcular_angulo_2d(ombro_esq, cotovelo_esq, pulso_esq)
            ang_dir = self.calcular_angulo_2d(ombro_dir, cotovelo_dir, pulso_dir)

            # 1. Montando o vetor de características EXATAMENTE como foi treinado na matriz
            features = [
                ang_esq, ang_dir,
                ombro_esq[0], ombro_esq[1], cotovelo_esq[0], cotovelo_esq[1], pulso_esq[0], pulso_esq[1],
                ombro_dir[0], ombro_dir[1], cotovelo_dir[0], cotovelo_dir[1], pulso_dir[0], pulso_dir[1]
            ]

            # 2. O VEREDITO DA IA: O modelo prevê o estado atual (0 ou 1) usando a álgebra do hiperplano
            estado_predito = self.modelo.predict([features])[0]

            # 3. A Máquina de Estados Autônoma
            if estado_predito == 1: # IA determinou que o atleta desceu
                if self.direcao == 0:
                    self.contador += 0.5
                    self.direcao = 1
            elif estado_predito == 0: # IA determinou que o atleta subiu
                if self.direcao == 1:
                    self.contador += 0.5
                    self.direcao = 0

            # --- DASHBOARD DE INFERÊNCIA (OTIMIZADO PARA VÍDEOS VERTICAIS E HORIZONTAIS) ---
            # Redução do container do contador para ganho de área útil lateral
            cv2.rectangle(img, (0, 0), (100, 100), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, str(int(self.contador)), (20, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 255), 4)
            
            # Identificação estática da postura atual rigidamente ancorada à direita do bloco azul (x=110)
            cv2.putText(img, "POSTURA: FLEXAO", (110, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
            
            # Feedback dinâmico da decisão do hiperplano ancorado e proporcional (x=110)
            texto_ia = "IA: DESCENDO" if estado_predito == 1 else "IA: SUBINDO"
            cor_ia = (0, 0, 255) if estado_predito == 1 else (255, 0, 0)
            cv2.putText(img, texto_ia, (110, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, cor_ia, 2)

        return img

# --- INÍCIO DA ESTEIRA DE TESTE (VÍDEO INÉDITO) ---
cap = cv2.VideoCapture('vid18.mp4')
motor = MotorInferencia()

print("Motor de Inferência (Deploy) iniciado. A IA assumiu o controle visual.")

while True:
    ret, img = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reinicia o vídeo para você auditar
        motor.contador = 0
        motor.direcao = 0
        continue

    # A blindagem geométrica da proporção original (Aspect Ratio)
    h, w = img.shape[:2]
    nova_altura = 700
    proporcao = nova_altura / h
    nova_largura = int(w * proporcao)
    img = cv2.resize(img, (nova_largura, nova_altura))

    img = motor.processar_frame(img)
    
    cv2.imshow('Projeto Fitness - Prova de Fogo', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()