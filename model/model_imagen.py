from keras.models import load_model
import numpy as np

# Cargar el modelo Keras
keras_model = load_model('modelo/mi_modelo_caso2.h5')


def predict_image(model, image):
    # Procesar la imagen como lo hiciste durante el entrenamiento
    processed_image = image.resize((128, 128))  # Asegúrate de que esto coincida con lo que hiciste en el entrenamiento
    processed_image = np.array(processed_image) / 255.0  # Normalización
    processed_image = processed_image.reshape(1, 128, 128, 3)  # Cambiar la forma para que coincida con las expectativas del modelo

    # Hacer la predicción
    prediction = model.predict(processed_image)
    return prediction


def predice(processed_image):
    print("================ PREDICT =========")
    # Hacer la predicción con la imagen
    prediction = predict_image(keras_model, processed_image)
    print(f"prediction: {prediction}")
    # Obtener la clase con la mayor probabilidad
    predicted_class = np.argmax(prediction)

    #prediction_result = int(prediction[0])
    print(f"PREDICCION: {predicted_class}")

    if predicted_class == 0:
        return "Fake News"
    elif predicted_class == 1:
        return "Not A Fake News"