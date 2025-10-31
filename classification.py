# predict_onnx.py
import onnxruntime as ort
import numpy as np


def classify(image,text,vectorizer, session, class_names):
    # Preprocess
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    text_features = vectorizer.transform([text]).toarray().astype(np.float32)
    
    # Predict with ONNX
    inputs = {
        session.get_inputs()[0].name: img_array,
        session.get_inputs()[1].name: text_features
    }
    
    predictions = session.run(None, inputs)[0][0]
    predicted_idx = int(np.argmax(predictions))
    
    return {
        'predicted_class': class_names[predicted_idx],
        'confidence': float(predictions[predicted_idx])
    }

