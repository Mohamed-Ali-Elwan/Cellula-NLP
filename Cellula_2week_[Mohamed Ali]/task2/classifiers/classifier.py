from abc import ABC, abstractmethod


class Classifier(ABC):

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def train(self, train_dataset, val_dataset):
        pass

    @abstractmethod
    def predict(self, text: str):
        pass

    @abstractmethod
    def save_model(self, path: str):
        pass

    @abstractmethod
    def load_saved_model(self, path: str):
        pass