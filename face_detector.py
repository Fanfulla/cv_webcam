"""
FaceDetector - Classe per rilevare volti nelle immagini usando DNN di openCV
"""

import cv2
import numpy as np

#Classe FACEDETECTOR --

class FaceDetector:

    """
    Rileva volti nelle immagini usando modello pre-addestrato caffe DNN
    """

    def __init__(self, prototxt_path, model_path, confidence_threshold=0.5):
        """
        Inizializza il detector caricando il modello
        Args: prototxt_path(str): Path al file
        model_path
        confidence_threshold
        """
        self.confidence_threshold = confidence_threshold

        print(f"[FaceDetector] Caricamento modello...")
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        print(f"[FaceDetector] Modello caricato con successo!")

    def detect_faces(self, image):
        """
        trova tutti i volti nell'immagine e returns una lista di dizionari con info sui volti
        """
        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image,
            1.0,
            (300,300),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7]* np.array([w,h,w,h])
                (x1,y1,x2,y2) = box.astype("int")
                box_w = x2 - x1
                box_h = y2 - y1

                faces.append({
                    "box": (x1, y1, box_w, box_h),
                    "confidence": float(confidence)
                })
                
        return faces


