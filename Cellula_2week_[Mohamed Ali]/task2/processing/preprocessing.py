import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Preprocessing:
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.label_encoder = LabelEncoder()

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        return self.data

    def remove_missing_values(self):
        self.data.dropna(inplace=True)

    def remove_duplicates(self):
        self.data.drop_duplicates(inplace=True)

    def clean_text(self, text: str) -> str:
        text = str(text).lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()
    
    def encode_labels(self):

        self.data["label"] = self.label_encoder.fit_transform(
            self.data["Toxic Category"]
        )

        return self.data
    
    def decode_label(self, prediction):

     return self.label_encoder.inverse_transform([prediction])[0]

    def preprocess_dataset(self):

        self.remove_missing_values()
        self.remove_duplicates()
        self.encode_labels()

        self.data["query"] = self.data["query"].apply(self.clean_text)
        self.data["image descriptions"] = self.data["image descriptions"].apply(self.clean_text)

        return self.data

    def split_dataset(self, test_size=0.2, random_state=42):
        train_df, test_df = train_test_split(
            self.data,
            test_size=test_size,
            random_state=random_state,
            stratify=self.data["Toxic Category"]
        )

        return train_df, test_df