"""
VideoProcessor
"""

import cv2

class VideoProcessor:
    """
    Processa video della webcam e applica la fd e age detection
    """

    def __init__(self, face_detector, age_estimator, gender_estimator):

        """
        si inizializza il video processor

        argomenti:
        face_detector
        age_estimator
        gender_estimator
        """

        self.face_detector = face_detector
        self.age_estimator = age_estimator
        self.gender_estimator = gender_estimator

        self.cap = None

    def process_frame(self, frame):
        """
        Processa un singolo frame: trova volti e stima età/genere
        
        Args:
            frame: Frame BGR da processare
            
        Returns:
            Frame con annotazioni disegnate
        """
        # STEP 1: Trova tutti i volti nel frame
        faces = self.face_detector.detect_faces(frame)
        
        # STEP 2: Per ogni volto trovato
        for face in faces:
            x, y, w, h = face['box']
            confidence = face['confidence']
            
            # Ritaglia il volto dall'immagine
            face_img = frame[y:y+h, x:x+w]
            
            # Verifica che il volto ritagliato non sia vuoto
            if face_img.size == 0:
                continue
            
            # STEP 3: Stima età e genere
            age = self.age_estimator.estimate_age(face_img)
            gender = self.gender_estimator.estimate_gender(face_img)
            
            # STEP 4: Disegna le informazioni sul frame
            self.draw_info(frame, x, y, w, h, age, gender, confidence)
        
        return frame

    def draw_info(self, frame, x, y, w, h, age, gender, confidence):

        """
        disegna i box con le info
        """

        color = (255, 0, 0) if gender == "Male" else (203, 192, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        text_line1 = f"{gender}, {age}"

        text_line2 = f"Conf: {confidence:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 0.6
        thickness = 2

        (text_w1, text_h1), _ = cv2.getTextSize(text_line1, font, font_scale, thickness)
        (text_w2, text_h2), _ = cv2.getTextSize(text_line2, font, font_scale, thickness)

        max_text_w = max(text_w1, text_w2)

        cv2.rectangle(frame,
                      (x, y - text_h1 - 25),
                      (x + max_text_w + 10, y-5),
                      (0, 0, 0), -1)

        cv2.putText(frame, text_line1, (x+5, y-10),
                    font, font_scale, color, thickness)

        cv2.putText(frame, text_line2, (x+5, y-10 - text_h1 - 5), font, 0.4, (255, 255, 255), 1)

    def start(self):

        """
        avvia la webcam
        """

        print("\n[VideoProcessor] AVVIO WEBCAM...")
        print("\nPremi 'q' per uscire\n")

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("ERRORE, non si avvia la cam")
            return

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("errore lettura frame")
                break

            processed_frame = self.process_frame(frame)

            cv2.imshow("Age & Gender Detector by SA", processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 = ESC
                break

        self.release()

    def release(self):
        """
        si rilasciano le risorse chiamate (cam e frame)
        """

        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n[VideoProcessor] WebcamChiusa - Ciao!")