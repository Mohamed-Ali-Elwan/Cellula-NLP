from processing.preprocessing import Preprocessing
from classifiers.distilbert_classifier import DistilBERTClassifier
from imagecaption import ImageCaption
from data.database_manager import DatabaseManager

from datasets import Dataset


class ClassificationService:

    def __init__(self, dataset_path):

        self.preprocessing = Preprocessing(dataset_path)

        self.preprocessing.load_data()

        self.preprocessing.preprocess_dataset()

        self.classifier = DistilBERTClassifier(num_labels=9)
        self.classifier.load_model()

        self.image_caption = ImageCaption()

        self.database = DatabaseManager()
    ########################################################
    # Train Model
    ########################################################

    def train_model(self):


        train_df, test_df = self.preprocessing.split_dataset()

        train_dataset = Dataset.from_pandas(
            train_df[["query", "label"]]
        )

        test_dataset = Dataset.from_pandas(
            test_df[["query", "label"]]
        )

        def tokenize(example):

            return self.classifier.tokenizer(

                example["query"],

                truncation=True,

                padding="max_length",

                max_length=128

            )

        train_dataset = train_dataset.map(tokenize)

        test_dataset = test_dataset.map(tokenize)

        train_dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "label"
            ]
        )

        test_dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "label"
            ]
        )

        self.classifier.train(

            train_dataset,

            test_dataset

        )

    ########################################################
    # Predict Text
    ########################################################

    def classify_text(self, text):
        prediction = self.classifier.predict(text)

        prediction_name = self.preprocessing.decode_label(prediction)

        self.database.save_record(
            text,
            prediction_name
        )

        return prediction_name



    ########################################################
    # Predict Image
    ########################################################

    def classify_image(self, image_path):

        caption = self.image_caption.generate_caption(image_path)

        prediction = self.classifier.predict(caption)

        prediction_name = self.preprocessing.decode_label(prediction)

        self.database.save_record(
         caption,
         prediction_name
        )

        return caption, prediction_name
    ########################################################
    # Database
    ########################################################

    def get_database(self):

        return self.database.get_all_records()