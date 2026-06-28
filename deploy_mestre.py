import cv2
import pickle
from cvzone.PoseModule import PoseDetector

print("Iniciando o Sistema Mestre de Reconhecimento Biomecânico...")

# 1. Carrega a Rede Neural Mestre Otimizada
with open('modelo_svm_olheiro.pkl', 'rb') as f:
    modelo_olheiro = pickle.load(f)

# 2. Inicializa o Tradutor Topologico
detector = PoseDetector(trackCon=0.70, detectionCon=0.70)

# Para testar ao vivo, deixe (0). Se quiser testar um video, troque 0 por "vid1.mp4", por exemplo.
cap = cv2.VideoCapture(0)

# Dicionario de Classes Mestre
classes = {
    1: ("FLEXAO DE BRACO", (255, 0, 0)),    # Azul
    2: ("ABDOMINAL", (0, 165, 255)),        # Laranja
    3: ("AGACHAMENTO", (0, 255, 0))         # Verde
}

while True:
    ret, img = cap.read()
    if not ret: break

    # Blindagem de Escala (Aspect Ratio)
    h, w = img.shape[:2]
    nova_altura = 700
    proporcao = nova_altura / h
    nova_largura = int(w * proporcao)
    img = cv2.resize(img, (nova_largura, nova_altura))

    # Extracao de Topologia do Corpo Inteiro
    img = detector.findPose(img, draw=True)
    lmList, bbox = detector.findPosition(img, draw=False, bboxWithHands=False)

    if len(lmList) != 0:
        # A exata matriz R^24 usada no treinamento
        pontos = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        linha_dados = []
        for p in pontos:
            linha_dados.extend(lmList[p][1:3])
        
        # 3. A Maquina decide qual exercicio está sendo executado no milissegundo atual
        predicao_mestre = modelo_olheiro.predict([linha_dados])[0]
        
        nome_exercicio, cor_exercicio = classes.get(predicao_mestre, ("DESCONHECIDO", (0,0,255)))

        # 4. Interface Visual do Cérebro
        cv2.rectangle(img, (0, 0), (nova_largura, 80), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, f"SISTEMA OLHEIRO ATIVO", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"POSTURA DETECTADA: {nome_exercicio}", (10, 65), cv2.FONT_HERSHEY_DUPLEX, 0.8, cor_exercicio, 2)

    cv2.imshow("Arquitetura Mestre (Nivel 1)", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()