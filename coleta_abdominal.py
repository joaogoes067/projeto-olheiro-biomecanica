import cv2
import numpy as np
import cvzone
from cvzone.PoseModule import PoseDetector
import csv
import os

class ExtratorAbdominal:
    def __init__(self, nome_arquivo_csv="dataset_abdominal.csv"):
        self.detector = PoseDetector(trackCon=0.70, detectionCon=0.70)
        self.frame_atual = 0
        self.nome_arquivo_csv = nome_arquivo_csv
        self.estado_anotacao = 0 # 0 = Repouso (Deitado), 1 = Contração (Em cima)
        self.iniciar_arquivo_csv()

    def iniciar_arquivo_csv(self):
        cabecalhos = [
            "video_origem", "frame", "ang_esq", "ang_dir", 
            "ombro_esq_x", "ombro_esq_y", "quadril_esq_x", "quadril_esq_y", "joelho_esq_x", "joelho_esq_y",
            "ombro_dir_x", "ombro_dir_y", "quadril_dir_x", "quadril_dir_y", "joelho_dir_x", "joelho_dir_y",
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
        return np.degrees(np.arccos(np.clip(cosseno_theta, -1.0, 1.0)))

    def processar_frame(self, img, nome_video):
        self.frame_atual += 1
        img = self.detector.findPose(img, draw=True)
        lmList, bbox = self.detector.findPosition(img, draw=False, bboxWithHands=False)

        if len(lmList) != 0:
            # MAPEAMENTO CIRÚRGICO: Ombro (11/12), Quadril (23/24), Joelho (25/26)
            ombro_esq = lmList[11][1:3]
            quadril_esq = lmList[23][1:3]
            joelho_esq = lmList[25][1:3]
            
            ombro_dir = lmList[12][1:3]
            quadril_dir = lmList[24][1:3]
            joelho_dir = lmList[26][1:3]

            # Ângulo da flexão do tronco
            ang_esq = self.calcular_angulo_2d(ombro_esq, quadril_esq, joelho_esq)
            ang_dir = self.calcular_angulo_2d(ombro_dir, quadril_dir, joelho_dir)

            # Gravação Tabular da nova geometria
            linha_dados = [
                nome_video, self.frame_atual, round(ang_esq, 2), round(ang_dir, 2),
                ombro_esq[0], ombro_esq[1], quadril_esq[0], quadril_esq[1], joelho_esq[0], joelho_esq[1],
                ombro_dir[0], ombro_dir[1], quadril_dir[0], quadril_dir[1], joelho_dir[0], joelho_dir[1],
                self.estado_anotacao
            ]
            self.salvar_linha_dados(linha_dados)

            # Destaque visual dos novos nós estruturais
            for p in [ombro_esq, quadril_esq, joelho_esq, ombro_dir, quadril_dir, joelho_dir]:
                cv2.circle(img, (p[0], p[1]), 10, (0, 255, 255), cv2.FILLED)

            # Interface do Supervisor
            cor_fundo = (0, 0, 255) if self.estado_anotacao == 1 else (255, 0, 0)
            texto_estado = "CONTRAÇÃO (1)" if self.estado_anotacao == 1 else "REPOUSO (0)"
            cv2.rectangle(img, (300, 400), (700, 480), cor_fundo, cv2.FILLED)
            cv2.putText(img, f"ABD: {texto_estado}", (340, 450), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)

        return img

# --- FILA DE PROCESSAMENTO ---
# Altere ou adicione os nomes dos vídeos de abdominal aqui
fila_de_videos = [f"abd{i}.mp4" for i in range(1, 18)]
extrator = ExtratorAbdominal()

print("Supervisão de Abdominais Iniciada.")
print("[ 1 ] -> Tronco levantado (Contração máxima).")
print("[ 0 ] -> Corpo deitado/estendido (Repouso).")
print("[ Q ] -> Pular vídeo.")

for nome_video in fila_de_videos:
    if not os.path.exists(nome_video):
        print(f"Aviso: {nome_video} não encontrado. Pulando.")
        continue
        
    cap = cv2.VideoCapture(nome_video)
    extrator.frame_atual = 0 
    extrator.estado_anotacao = 0 

    while True:
        ret, img = cap.read()
        if not ret: break 

        # Preservação da Proporção Original (Aspect Ratio)
        h, w = img.shape[:2]
        nova_altura = 700
        proporcao = nova_altura / h
        nova_largura = int(w * proporcao)
        img = cv2.resize(img, (nova_largura, nova_altura))
        cvzone.putTextRect(img, f'Anotando: {nome_video}', [20, 30], thickness=2, border=2, scale=2)
        
        
        img = extrator.processar_frame(img, nome_video)
        cv2.imshow('Laboratorio de IA - Dataset Abdominal', img)
        
        key = cv2.waitKey(60) & 0xFF
        if key == ord('1'):
            extrator.estado_anotacao = 1
        elif key == ord('0'):
            extrator.estado_anotacao = 0
        elif key == ord('q'):
            break

    cap.release()

cv2.destroyAllWindows()
print("\nDataset de abdominais gerado com sucesso!")