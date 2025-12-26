"""
AGE e GENDER DETECTOR
"""

import age_estimator
from face_detector import FaceDetector
from age_estimator import AgeEstimator
import face_detector
from gender_estimator import GenderEstimator
import gender_estimator
from video_processor import VideoProcessor


def main():
    """
    funzione di avvio
    """

    print("="*60)
    print("AGE e GENDER DETECTOR by SA")
    print("="*60)

    FACE_PROTOTXT = "models/deploy.prototxt"

    FACE_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"

    AGE_PROTOTXT = "models/age_deploy.prototxt"
    AGE_MODEL = "models/age_net.caffemodel"

    GENDER_PROTOTXT = "models/gender_deploy.prototxt"
    GENDER_MODEL = "models/gender_net.caffemodel"

    print("\nCaricamento Modelli in corso...")


    face_detector = FaceDetector(FACE_PROTOTXT, FACE_MODEL, confidence_threshold=0.5)
    age_estimator = AgeEstimator(AGE_PROTOTXT, AGE_MODEL)
    gender_estimator = GenderEstimator(GENDER_PROTOTXT, GENDER_MODEL)

    print("\nTutti i modelli caricati con successo!")

    processor = VideoProcessor(face_detector, age_estimator, gender_estimator)

    processor.start()

if __name__ == "__main__":
    main()


