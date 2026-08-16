from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

import pandas as pd
import shutil
import os

from llm import LLM
from whisper_service import WhisperService
from database import SQLiteDatabase
from sql_generator import SQLGenerator
from sql_validator import SQLValidator
from analysis_service import AnalysisService


app = FastAPI(
    title="Voice Data Analysis API"
)


# ==========================================
# INITIALIZE COMPONENTS
# ==========================================

llm = LLM.get_llm1()

whisper = WhisperService()

database = SQLiteDatabase()

sql_generator = SQLGenerator(llm)

sql_validator = SQLValidator()

analysis_service = AnalysisService(
    database=database,
    sql_generator=sql_generator,
    sql_validator=sql_validator
)


# ==========================================
# HEALTH
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "running"
    }


# ==========================================
# DATASET UPLOAD
# ==========================================

@app.post("/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...)
):

    file_path = f"data/{file.filename}"

    os.makedirs(
        "data",
        exist_ok=True
    )

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    if file.filename.endswith(".csv"):

        dataframe = pd.read_csv(
            file_path
        )

    elif file.filename.endswith(".xlsx"):

        dataframe = pd.read_excel(
            file_path
        )

    else:

        raise HTTPException(
            status_code=400,
            detail="Only CSV and Excel files are supported."
        )

    database.load_dataframe(
        dataframe
    )

    return {
        "message": "Dataset uploaded successfully.",
        "rows": len(dataframe),
        "columns": list(dataframe.columns)
    }


# ==========================================
# TEXT ANALYSIS
# ==========================================

@app.post("/analyze")
async def analyze(
    question: str
):

    try:

        result = analysis_service.analyze(
            question
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# ==========================================
# VOICE ANALYSIS
# ==========================================

@app.post("/voice")
async def voice_analysis(
    file: UploadFile = File(...)
):

    audio_path = f"data/{file.filename}"

    os.makedirs(
        "data",
        exist_ok=True
    )

    with open(
        audio_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    transcription = whisper.transcribe(
        audio_path
    )

    question = transcription["text"]

    result = analysis_service.analyze(
        question
    )

    return {
        "language": transcription["language"],
        "transcription": question,
        "sql": result["sql"],
        "result": result["result"]
    }