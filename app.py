from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import uvicorn
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import time
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from summary_llm import summarize
from sqlite_db import SQLiteDB
from milvus_db import MilvusDB
from paddleocr import PaddleOCR
load_dotenv(override=True)
from logger.logger_config import Logger
from config import MILVUS_COLLECTION
from dotenv import load_dotenv
import os
log = Logger.get_logger(__name__)
load_dotenv(override=True)

app = FastAPI(title="Document Processing API")

# Initialize databases
sqlite_db = SQLiteDB(os.getenv("SQLITE_DB_PATH", "documents.db"))
sqlite_db.create_tables()

milvus_db = MilvusDB()
milvus_collection = MILVUS_COLLECTION
milvus_db.create_db(milvus_collection)

#ocr model
ocr_model = PaddleOCR(lang='en', use_angle_cls=False, det_limit_side_len=960)

processor = DocumentProcessor(ocr_model=ocr_model, sqlite_db=sqlite_db, milvus_db=milvus_db)


def read_image(file: UploadFile) -> np.ndarray:
    """Read uploaded image file into OpenCV numpy array."""
    try:
        image_bytes = file.file.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


@app.post("/process_document/")
async def process_document(
    file: UploadFile = File(...),
    
):
    
    try:
        image = read_image(file)
        result = processor.process(image=image)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

@app.post("/summarize_document/")
async def summarize_document(
    file: UploadFile = File(...),
    summarize_doc: Optional[bool] = Form(True)
):

    try:
        image = read_image(file)
        result = processor.process(image=image, summarize_doc=summarize_doc)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
