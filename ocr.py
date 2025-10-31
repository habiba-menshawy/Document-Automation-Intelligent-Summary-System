def extract_text(img, ocr=None )-> str:
        """
        Extract text from document.
        
        Args:
            file_path: Path to the document file
            preprocess: Whether to apply preprocessing
            
        Returns:
            List of dictionaries containing extracted text and metadata for each page
        """
       
        # Perform OCR
        ocr_result = ocr.predict(img)
        # Combine all text
        full_text = '\n'.join([line for line in ocr_result[0]["rec_texts"]])
        return full_text