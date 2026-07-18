
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Data -------------------------------------------------------------
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "cellula toxic data  (1).csv")
DATABASE_PATH = os.path.join(BASE_DIR, "data", "database.csv")

# --- Text classifier (DistilBERT + LoRA) -------------------------------
# Directory produced by DistilBERTClassifier.save_model() after training.
DISTILBERT_MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

# Number of unique classes in the "Toxic Category" column of the dataset.
# (Safe, Violent Crimes, Non-Violent Crimes, unsafe, Unknown S-Type,
#  Sex-Related Crimes, Suicide & Self-Harm, Elections,
#  Child Sexual Exploitation)
NUM_LABELS = 9

# --- Image captioning (BLIP) -------------------------------------------
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"