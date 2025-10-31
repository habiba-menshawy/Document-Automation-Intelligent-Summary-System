from tensorflow import keras

model = keras.models.load_model("saved_models_hybrid/best_hybrid_model.keras")
model.export("saved_models_hybrid/model_savedmodel")  # new in TF 2.13+
