from ultralytics import YOLO
import cv2

# Charger le modèle YOLOv8
model = YOLO('yolov8n.pt')  # 'yolov8n.pt' pour la version nano

# Capturer la vidéo
cap = cv2.VideoCapture(0)  # 0 pour webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Détection des objets
    results = model(frame)
    
    # Affichage des résultats
    annotated_frame = results[0].plot()
    
    cv2.imshow("YOLOv8 Détection", annotated_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
