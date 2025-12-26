"""
GenderEstimator - Maschio o Femmina
"""

import cv2
import numpy as np


class GenderEstimator:
    """
    stimiamo il genere di un volto
    """

    GENDERS = ["Male", "Female"]

    def __init__(self, prototxt_path, model_path):
        """
        Inizializziamo il gender estimator
        """

        print("f[GenderEstimator] Caricamento del modello...")

        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        print(f"[GenderEstimator] Modello caricato con successo!")

    def estimate_gender(self, face_image):

        """
        Stima il genere di un volto
        """

        blob = cv2.dnn.blobFromImage(
            face_image,
            1.0,
            (227, 227),
            (78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False
        )


        self.net.setInput(blob)

        gender_preds = self.net.forward()

        gender_index = gender_preds[0].argmax()

        gender = self.GENDERS[gender_index]

        return gender
