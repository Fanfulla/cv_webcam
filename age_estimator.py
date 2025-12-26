"""
Age-estimator
"""

import cv2
import numpy as np

class AgeEstimator:
    """
    Stimiamo l'eta di una persona con questa classe
    """

    AGE_RANGES= [
                "(0-2)", "(4-6)", "(8-12)", "(15-20)",
        "(25-32)", "(38-43)", "(48-53)", "(60-100)"
    ]

    def __init__(self, prototxt_path, model_path):
        """
        Inizializza l'age estimator:

        Args: prototxt_path(str) path al file age_deply_prototext
        model_path (str): Path al file age_net
        """

        print(f"[AgeEstimator] Caricamento Modello...")
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        print(f"[AgeEstimator] Modello caricato con successo")

    def estimate_age(self, face_image):
        """
        Stimare l'età di un volto
        """

        blob = cv2.dnn.blobFromImage(
            face_image,
            1.0,
            (227, 227),
            (78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False
        )

        self.net.setInput(blob)

        age_preds = self.net.forward()

        age_index = age_preds[0].argmax()

        age_range = self.AGE_RANGES[age_index]
        return age_range
