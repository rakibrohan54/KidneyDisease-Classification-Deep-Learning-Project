import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        # Define class labels
        self.class_labels = {
            0: 'Normal',
            1: 'Tumor',
        }

    def predict(self):
        # Load the model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        # Load the image to be predicted
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Normalize the image
        test_image = test_image / 255.0

        # Make the prediction
        result = model.predict(test_image)
        predicted_class = np.argmax(result, axis=1)[0]

        # Map the result to the corresponding class label
        prediction = self.class_labels.get(predicted_class, "Unknown")

        return [{"image": prediction}]

# Example usage:
# prediction_pipeline = PredictionPipeline("path_to_image.jpg")
# prediction_result = prediction_pipeline.predict()
# print(prediction_result)
