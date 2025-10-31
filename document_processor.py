from preprocessing import ImagePreprocessor
from ocr import extract_text
from classification import classify
from entity_extraction import NERExtractor
import cv2
import os
import json
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import onnxruntime as ort
import numpy as np
from PIL import Image
import joblib
from dotenv import load_dotenv
from sqlite_db import SQLiteDB
from milvus_db import MilvusDB
import time
import ollama
from config import OllamaConfig, MILVUS_COLLECTION
from sqlite_db import SQLiteDB
import numpy as np
load_dotenv(override=True)
from logger.logger_config import Logger

log = Logger.get_logger(__name__)

class DocumentProcessor:
    """
    A unified class that processes document images:
    - Preprocesses image
    - Extracts text using OCR
    - Classifies document type
    - Extracts entities from text
    """

    def __init__(self, ocr_model=None,sqlite_db=SQLiteDB(os.getenv("SQLITE_DB_PATH", "documents.db")),milvus_db=MilvusDB()):
        """
        Initialize the document processor.
        Args:
            image (np.ndarray): Input image as a numpy array.
            ocr_model: Optional OCR model instance.
        """
        self.ocr_model = ocr_model
        if self.ocr_model is None:
            self.ocr_model = PaddleOCR(lang='en', use_angle_cls=False, det_limit_side_len=960)
        self.session = ort.InferenceSession(os.getenv("CLASSIFICATION_MODEL_PATH"))
        self.vectorizer = joblib.load(os.getenv("CLASSIFICATION_VECTORIZER_PATH"))
        self.class_names = joblib.load(os.getenv("CLASSIFICATION_CLASS_NAMES_PATH"))
        self.extractor = NERExtractor()
        self.sqlite_db = sqlite_db
        self.milvus_db = milvus_db

    def preprocess(self,image):
        """Apply preprocessing steps on the image."""
        preprocessor = ImagePreprocessor(image)
        preprocessor.rotate_image()
        preprocessor.bilateral_filter()
        preprocessor.clahe()
        preprocessor.remove_border_adaptive()
        preprocessed_image = preprocessor.get_image()
        return preprocessed_image

    def run_ocr(self,image):
        """Extract text from the preprocessed image."""
        if image is None:
            raise ValueError("Image not preprocessed. Call preprocess() first.")
        extracted_text = extract_text(image,ocr=self.ocr_model)
        return extracted_text

    def run_classification(self,image=None, extracted_text=''):
        """Classify the document based on its image and text content."""
        if image is None or extracted_text is None:
            raise ValueError("Preprocessing or OCR must be completed first.")
        
        # Convert from OpenCV (BGR) → PIL (RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        classification_result = classify(image=pil_image, text=extracted_text, vectorizer=self.vectorizer, session=self.session, class_names=self.class_names)
        return classification_result

    def extract_entities(self, entity_type: str, extracted_text: str = None):
        """Extract named entities from the text."""
        if extracted_text is None:
            raise ValueError("OCR must be performed before entity extraction.")
        
        entities = self.extractor.extract_entities(extracted_text, entity_type)
        
        return entities

    def process(self,  image: np.ndarray = None, summarize_doc: bool = False):
        """
        Full pipeline: preprocess → OCR → classify → extract entities.
        Saves results of each step.
        """
        # 1- process the image
        log.info("Starting preprocessing...")
        preprocessed_image = self.preprocess(image)

        log.info("Running OCR...")
        text = self.run_ocr(preprocessed_image)

        log.info("Classifying document...")
        classification = self.run_classification(image=preprocessed_image, extracted_text=text )

        log.info(f"Extracting  entities for document type {classification['predicted_class']}...")
        entities = self.extract_entities(classification["predicted_class"], extracted_text=text)

        log.info("Processing complete.")

        # 2. Generate embedding
        try:
            log.info("Generating embedding...")
            embedding_vector = np.array(ollama.embed(model=OllamaConfig.embedding_model, input=text)['embeddings'][0])
        except Exception as e:
            log.error(f"Failed to generate embedding: {e}")
            embedding_vector = np.random.rand(768).tolist()

        # 3. Insert into Milvus first
        doc_id = None
        try:
            log.info("Inserting into Milvus...")
            milvus_data = {
                "embedding": embedding_vector,
                "created_at": int(time.time()),
                "text": text
            }
            insert_result = self.milvus_db.insert(MILVUS_COLLECTION, milvus_data)
            self.milvus_db.flush(MILVUS_COLLECTION)
            doc_id = insert_result.get('ids')[0]  # Get the Milvus PK
            log.info(f"Inserted into Milvus with PK={doc_id}")
        except Exception as e:
            log.error(f"Milvus insertion failed: {e}")
            raise RuntimeError(f"Milvus insertion failed: {e}")

        # 4. Optional summarization
        summary_text = None
        if summarize_doc:
            log.info("Generating summary...")
            try:
                from summary_llm import summarize  # import your summarize function
                summary_text = summarize({
                    "ocr_text": text,
                    "classification": classification,
                    "entities": entities
                })
            except Exception as e:
                log.error(f"Summarization failed: {e}")
                summary_text = None

        # 5. Insert into SQLite
        try:
            if summarize_doc:
                    doc_query = """
                        INSERT INTO documents (id, filename, summary, date, classification)
                        VALUES (?, ?, ?, ?, ?)
                    """
                    doc_params = (
                        doc_id,
                        f"document_{doc_id}.png",
                        summary_text,
                        time.strftime("%Y-%m-%d"),
                        classification["predicted_class"]
                    )
                    self.sqlite_db.execute_update(doc_query, doc_params)

            # Insert entities
            if entities:
                entity_query = """
                    INSERT INTO entities (document_id, label, text, start_pos, end_pos, confidence, method)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                entity_params = [
                    (
                        doc_id,
                        e.get("label"),
                        e.get("text"),
                        e.get("start"),
                        e.get("end"),
                        e.get("confidence"),
                        e.get("method")
                    ) for e in entities
                ]
                self.sqlite_db.execute_many(entity_query, entity_params)

        except Exception as e:
            log.error(f"SQLite insertion failed: {e}")
            raise RuntimeError(f"SQLite insertion failed: {e}")

        return {
            # "preprocessed_image": preprocessed_image,
            "ocr_text": text,
            "classification": classification,
            "entities": entities,
            "summary": summary_text,
            "doc_id": doc_id,
        }
if __name__ == "__main__":
    img = cv2.imread("../archive/dataset/Email/2064213021d.jpg")
    processor = DocumentProcessor()
    results = processor.process(image=img, summarize_doc=True)

    print("\nExtracted Text:\n", results["ocr_text"])
    print("\nClassification:\n", results["classification"])
    print("\nEntities:\n", results["entities"])
    print("\nSummary:\n", results["summary"])
    print("\nDocument ID:\n", results["doc_id"])
