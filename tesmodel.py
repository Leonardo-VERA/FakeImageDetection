from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Charger le modele cree
model = load_model('fake_image_detector.h5')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Path image a predire
img_path = 'C:\AQUI TODO\python_dir\data\Au\Au_ani_00002.jpg'

# Preprocessing image
img_array = preprocess_image(img_path)

# prediction
prediction = model.predict(img_array)
print(f"Prediction: {prediction[0][0]}")

# interpretation prediction
if prediction[0][0] > 0.5:
    print("L'imagen est manipulé.")
else:
    print("L'image est authentique.")
