

import pandas as pd

import config
from processing.preprocessing import Preprocessing
from data.dataset import ToxicDataset
from classifiers.distilbert_classifier import DistilBERTClassifier


def build_text_label_frame(data: pd.DataFrame) -> pd.DataFrame:
   
    query_rows = data[["query", "label"]].rename(columns={"query": "text"})
    caption_rows = data[["image descriptions", "label"]].rename(
        columns={"image descriptions": "text"}
    )
    combined = pd.concat([query_rows, caption_rows], ignore_index=True)
    return combined


def main():
    print("Loading and cleaning data...")
    preprocessing = Preprocessing(config.RAW_DATA_PATH)
    preprocessing.load_data()
    preprocessing.preprocess_dataset()

    train_df, val_df = preprocessing.split_dataset(test_size=0.2, random_state=42)

    train_frame = build_text_label_frame(train_df)
    val_frame = build_text_label_frame(val_df)

    print(f"Train examples: {len(train_frame)} | Val examples: {len(val_frame)}")

    print("Loading base DistilBERT + LoRA...")
    classifier = DistilBERTClassifier(num_labels=config.NUM_LABELS)
    classifier.load_model()

    train_dataset = ToxicDataset(
        texts=train_frame["text"].tolist(),
        labels=train_frame["label"].tolist(),
        tokenizer=classifier.tokenizer,
    )
    val_dataset = ToxicDataset(
        texts=val_frame["text"].tolist(),
        labels=val_frame["label"].tolist(),
        tokenizer=classifier.tokenizer,
    )

    classifier.train(
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        output_dir=config.DISTILBERT_MODEL_DIR,
        epochs=3,
        batch_size=16,
        learning_rate=2e-5,
    )

    print(f"Done. Model saved to: {config.DISTILBERT_MODEL_DIR}")


if __name__ == "__main__":
    main()