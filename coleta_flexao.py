import cv2
import numpy as np
import cvzone
from cvzone.PoseModule import PoseDetector
import csv
import os

class AnalisadorFlexao:
    def __init__(self, nome_arquivo_csv="dataset_flexao.csv"):
        self.detector = PoseDetector(trackCon=0.70, detectionCon=0.70)
        self.frame_atual = 0
        self.nome_arquivo_csv = nome_arquivo_csv
        self.estado_anotacao = 0 # 0 = Repouso, 1 = Contração
        self.iniciar_arquivo_csv()

    def iniciar_arquivo_csv(self):
        cabecalhos = [
            "video_origem", "frame", "ang_esq", "ang_dir", 
            "ombro_esq_x", "ombro_esq_y", "cotovelo_esq_x", "cotovelo_esq_y", "pulso_esq_x", "pulso_esq_y",
            "ombro_dir_x", "ombro_dir_y", "cotovelo_dir_x", "cotovelo_dir_y", "pulso_dir_x", "pulso_dir_y",
            "label"
        ]
        if not os.path.exists(self.nome_arquivo_csv):
            with open(self.nome_arquivo_csv, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(cabecalhos)

    def salvar_linha_dados(self, dados):
        with open(self.nome_arquivo_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(dados)
            
    @staticmethod
    def calcular_angulo_2d(p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: return 0
        cosseno_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        angulo_rad = np.arccos(np.clip(cosseno_theta, -1.0, 1.0))
        return np.degrees(angulo_rad)

    def processar_frame(self, img, nome_video):
        self.frame_atual += 1
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

            # --- GRAVAÇÃO PURA COM A SUA SUPERVISÃO ---
            linha_dados = [
                nome_video, self.frame_atual, round(ang_esq, 2), round(ang_dir, 2),
                ombro_esq[0], ombro_esq[1], cotovelo_esq[0], cotovelo_esq[1], pulso_esq[0], pulso_esq[1],
                ombro_dir[0], ombro_dir[1], cotovelo_dir[0], cotovelo_dir[1], pulso_dir[0], pulso_dir[1],
                self.estado_anotacao # Esse número vem do seu teclado
            ]
            self.salvar_linha_dados(linha_dados)

            # Desenhando pontos puros
            for p in [ombro_esq, cotovelo_esq, pulso_esq, ombro_dir, cotovelo_dir, pulso_dir]:
                cv2.circle(img, (p[0], p[1]), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (p[0], p[1]), 15, (0, 255, 0), 5)

            cv2.putText(img, f"Ang R: {int(ang_dir)}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)
            cv2.putText(img, f"Ang L: {int(ang_esq)}", (750, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)

            # INTERFACE DO SUPERVISOR
            cor_fundo = (0, 0, 255) if self.estado_anotacao == 1 else (255, 0, 0)
            texto_estado = "DESCENDO (1)" if self.estado_anotacao == 1 else "REPOUSO (0)"
            cv2.rectangle(img, (300, 400), (700, 480), cor_fundo, cv2.FILLED)
            cv2.putText(img, f"ROTULO: {texto_estado}", (320, 450), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)

        return img

# --- MOTOR PRINCIPAL EM LOTE (ANOTAÇÃO MANUAL) ---
fila_de_videos = [f"vid{i}.mp4" for i in range(35, 57)] 
analisador = AnalisadorFlexao()

print("Motor de Anotação Assistida Iniciado.")
print("=== TECLAS DE COMANDO ===")
print("[ 1 ] -> Ativar estado de DESCIDA/CONTRAÇÃO.")
print("[ 0 ] -> Ativar estado de SUBIDA/REPOUSO.")
print("[ Q ] -> Pular o vídeo atual.")

for nome_video in fila_de_videos:
    if not os.path.exists(nome_video):
        continue
        
    cap = cv2.VideoCapture(nome_video)
    analisador.frame_atual = 0 
    analisador.estado_anotacao = 0 # Garante que todo vídeo começa no zero

    while True:
        ret, img = cap.read()
        if not ret: break 

        h, w = img.shape[:2]
        nova_altura = 700
        proporcao = nova_altura / h
        nova_largura = int(w * proporcao)
        img = cv2.resize(img, (nova_largura, nova_altura))
        cvzone.putTextRect(img, f'Anotando: {nome_video}', [20, 30], thickness=2, border=2, scale=2)
        
        img = analisador.processar_frame(img, nome_video)
        cv2.imshow('Projeto Fitness - Esteira de ML', img)
        
        # Aumentei o tempo de delay para 60ms. O vídeo rodará em câmera lenta para você não errar o clique.
        key = cv2.waitKey(60) & 0xFF
        
        if key == ord('1'):
            analisador.estado_anotacao = 1
        elif key == ord('0'):
            analisador.estado_anotacao = 0
        elif key == ord('q'):
            break

    cap.release()

cv2.destroyAllWindows()
print("\nDataset anotado com perfeição humana.")