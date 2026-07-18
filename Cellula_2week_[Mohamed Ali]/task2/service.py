
from classifiers.distilbert_classifier import DistilBERTClassifier
from imagecaption import ImageCaption
from processing.preprocessing import Preprocessing
from data.database_manager import DatabaseManager


class ToxicityService:

    def __init__(self, preprocessing: Preprocessing, classifier: DistilBERTClassifier,
                 captioner: ImageCaption, db: DatabaseManager):
        self.preprocessing = preprocessing
        self.classifier = classifier
        self.captioner = captioner
        self.db = db

    def _predict_label(self, text: str) -> str:
        cleaned = self.preprocessing.clean_text(text)
        prediction_id = self.classifier.predict(cleaned)
        return self.preprocessing.decode_label(prediction_id)

    def classify_text(self, text: str) -> str:
        label = self._predict_label(text)
        self.db.save_record(user_input=text, prediction=label)
        return label

    def classify_image(self, image) -> tuple:
        caption = self.captioner.generate_caption(image)
        label = self._predict_label(caption)
        self.db.save_record(user_input=caption, prediction=label)
        return caption, label

    def get_history(self):
        return self.db.get_all_records()