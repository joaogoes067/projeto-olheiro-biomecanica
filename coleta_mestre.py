import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector
import csv
import os

class ExtratorMestre:
    def __init__(self, nome_arquivo_csv="dataset_olheiro.csv"):
        self.detector = PoseDetector(trackCon=0.70, detectionCon=0.70)
        self.nome_arquivo_csv = nome_arquivo_csv
        self.iniciar_arquivo_csv()

    def iniciar_arquivo_csv(self):
        # A Nova Matriz Universal em R^24
        cabecalhos = [
            "video_origem", 
            "ombro_esq_x", "ombro_esq_y", "ombro_dir_x", "ombro_dir_y",
            "cotovelo_esq_x", "cotovelo_esq_y", "cotovelo_dir_x", "cotovelo_dir_y",
            "pulso_esq_x", "pulso_esq_y", "pulso_dir_x", "pulso_dir_y",
            "quadril_esq_x", "quadril_esq_y", "quadril_dir_x", "quadril_dir_y",
            "joelho_esq_x", "joelho_esq_y", "joelho_dir_x", "joelho_dir_y",
            "tornozelo_esq_x", "tornozelo_esq_y", "tornozelo_dir_x", "tornozelo_dir_y",
            "label_exercicio" # 1 = Flexao, 2 = Abdominal, 3 = Agachamento
        ]
        if not os.path.exists(self.nome_arquivo_csv):
            with open(self.nome_arquivo_csv, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(cabecalhos)

    def processar_video(self, nome_video, label):
        if not os.path.exists(nome_video):
            return
        
        cap = cv2.VideoCapture(nome_video)
        print(f"Extraindo topologia de corpo inteiro: {nome_video} -> Label {label}")
        
        while True:
            ret, img = cap.read()
            if not ret: break

            # Nossa blindagem de Aspect Ratio
            h, w = img.shape[:2]
            nova_altura = 700
            proporcao = nova_altura / h
            nova_largura = int(w * proporcao)
            img = cv2.resize(img, (nova_largura, nova_altura))

            img = self.detector.findPose(img, draw=False)
            lmList, bbox = self.detector.findPosition(img, draw=False, bboxWithHands=False)

            if len(lmList) != 0:
                # Mapeamento Global: Ombros, Cotovelos, Pulsos, Quadril, Joelhos, Tornozelos
                pontos = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
                linha_dados = [nome_video]
                
                for p in pontos:
                    linha_dados.extend(lmList[p][1:3]) # Extrai X e Y
                
                linha_dados.append(label)
                
                with open(self.nome_arquivo_csv, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(linha_dados)

        cap.release()

# --- A AUTOMACAO DA NOSSA BIBLIOTECA ---
extrator = ExtratorMestre()

# O código deduz a resposta pelo nome do arquivo
videos_flexao = [f"vid{i}.mp4" for i in range(1, 57)]      # Label 1
videos_abdominal = [f"abd{i}.mp4" for i in range(1, 18)]   # Label 2
videos_agacha = [f"agacha{i}.mp4" for i in range(1, 44)]   # Label 3

print("Iniciando mineracao massiva para o Olheiro...")
print("Descanse. A maquina assumiu o controle absoluto.")

for v in videos_flexao: extrator.processar_video(v, 1)
for v in videos_abdominal: extrator.processar_video(v, 2)
for v in videos_agacha: extrator.processar_video(v, 3)

print("\nMatriz universal gerada com sucesso! O 'dataset_olheiro.csv' nasceu.")