import torch
import numpy as np

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


from classifiers.classifier import Classifier


class DistilBERTClassifier(Classifier):

    def __init__(self, num_labels):

        super().__init__("distilbert-base-uncased")

        self.num_labels = num_labels

    ##################################################
    # Load DistilBERT + LoRA
    ##################################################

    def load_model(self):

        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.model_name
        )

        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )

        lora_config = LoraConfig(

            task_type=TaskType.SEQ_CLS,

            r=16,

            lora_alpha=32,

            lora_dropout=0.1,

            bias="none",

            target_modules=[
                "q_lin",
                "v_lin"
            ]
        )

        self.model = get_peft_model(
            self.model,
            lora_config
        )

        self.model.print_trainable_parameters()

    ##################################################
    # Metrics
    ##################################################

    def compute_metrics(self, eval_pred):

        logits, labels = eval_pred

        predictions = np.argmax(
            logits,
            axis=1
        )

        precision, recall, f1, _ = precision_recall_fscore_support(

            labels,

            predictions,

            average="weighted", zero_division=0

        )

        accuracy = accuracy_score(
            labels,
            predictions
        )

        return {

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1

        }

    ##################################################
    # Train
    ##################################################

    def train(

        self,

        train_dataset,

        validation_dataset,

        output_dir="saved_model",

        epochs=3,

        batch_size=16,

        learning_rate=2e-5

    ):

        training_args = TrainingArguments(

            output_dir=output_dir,

            eval_strategy="epoch",

            save_strategy="epoch",

            logging_strategy="epoch",

            learning_rate=learning_rate,

            per_device_train_batch_size=batch_size,

            per_device_eval_batch_size=batch_size,

            num_train_epochs=epochs,

            weight_decay=0.01,

            load_best_model_at_end=True,

            metric_for_best_model="f1",

            greater_is_better=True,

            report_to="none"

        )

        trainer = Trainer(

            model=self.model,

            args=training_args,

            train_dataset=train_dataset,

            eval_dataset=validation_dataset,

            processing_class=self.tokenizer,

            compute_metrics=self.compute_metrics

        )

        trainer.train()

        trainer.save_model(output_dir)

        self.tokenizer.save_pretrained(output_dir)

    ##################################################
    # Predict
    ##################################################

    def predict(self, text):

        self.model.eval()

        inputs = self.tokenizer(

            text,

            truncation=True,

            padding=True,

            max_length=128,

            return_tensors="pt"

        )

        with torch.no_grad():

            outputs = self.model(

                input_ids=inputs["input_ids"],

                attention_mask=inputs["attention_mask"]

            )

        prediction = torch.argmax(

            outputs.logits,

            dim=1

        )

        return prediction.item()

    ##################################################
    # Save
    ##################################################

    def save_model(self, path):

        self.model.save_pretrained(path)

        self.tokenizer.save_pretrained(path)

    ##################################################
    # Load Saved Model
    ##################################################

    def load_saved_model(self, path):

        self.tokenizer = DistilBertTokenizer.from_pretrained(path)

        base_model = DistilBertForSequenceClassification.from_pretrained(

            "distilbert-base-uncased",

            num_labels=self.num_labels

        )

        self.model = PeftModel.from_pretrained(

            base_model,

            path

        )