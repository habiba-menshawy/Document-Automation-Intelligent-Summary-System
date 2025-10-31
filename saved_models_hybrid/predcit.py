# predict_onnx.py
import onnxruntime as ort
import numpy as np
from PIL import Image
import joblib

# Load ONNX model
session = ort.InferenceSession('saved_models_hybrid/model.onnx')
vectorizer = joblib.load('saved_models_hybrid/tfidf_vectorizer.pkl')
class_names = joblib.load('saved_models_hybrid/class_names.pkl')

def classify(image_path,text):
    # Preprocess
    img = Image.open(image_path).convert('RGB').resize((224, 224))
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



# Example usage
if __name__ == "__main__":
    result = classify(
        "/mnt/c/THE TASK/archive/dataset/Report/569669.jpg",
        """USE OF COTININE BLOOD CONCENTRATION
AS A DETECTION METHOD
OF NICOTINE DELIVERY WITH CIGARETTE SMOKING
PART II. DISTRIBUTION OF AVERAGE VALUES FOR NICOTINE DELIVERY
PER CIGARETTE SMOKER
T. D. Darby, Ph.D.
and
James E. McNamee, Ph.D.
University of South Carolina School of Medicine
These evaluations of cotinine plasma values from a study of thirty-nine
smokers of ultra-low nicotine and tar cigarettes conducted by Franklin Institute
provide further assurance that the Barclay brand is an ultra-low delivery cigarette.
Several methods have been used to date that provide for classification of different
brands according to the nicotine and tar content delivered to the smoker. The FTC
method is the standard procedure for establishing nicotine and tar delivery for
each cigarette smoked.
The use of cotinine plasma level determinations obtained from smokers under
established smoking conditions allows for comparison between predicted values for
plasma cotinine and observed values. Using the kinetic equations described in
Part I and the sequential solution computer model, it is possible to arrive at a
predicted plasma level for cotinine based upon cigarette consumption utilizing
labeled value for nicotine delivery. With establishment of the mathematical
501026597
relationship between observed and predicted continine values, it is possible to
estimate the average delivery for each cigarette smoked. Figures 1 through 4
provide a histogram of the average delivery for the brands studies. Figures 5
and 6 illustrate the variation in number of cigarettes smoked per day for the
different brands studied. The diary values were used to construct these histograms.
Since smoking patterns varied within days for each smoker, it is possible that
these values may not reflect exact values for use with the computer model."""
    )
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
