import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Definir routes
dataset_dir = r'C:\AQUI TODO\python_dir\data'
output_dir = r'C:\AQUI TODO\python_dir\casia2_processed'
os.makedirs(output_dir, exist_ok=True)

# Creer des dossier entrainement y validation
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Creer sousdossiers
for category in ['Au', 'Tp']:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)

# Diviser les imágenes en groupes d'entrainement et validation
for category in ['Au', 'Tp']:
    category_path = os.path.join(dataset_dir, category)
    images = os.listdir(category_path)
    train_images, val_images = train_test_split(images, 
                                                test_size=0.2, 
                                                random_state=42)
    
    for image in train_images:
        shutil.copy(os.path.join(category_path, image), os.path.join(train_dir, category, image))
    
    for image in val_images:
        shutil.copy(os.path.join(category_path, image), os.path.join(val_dir, category, image))

# Definir generateurs de donnees
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   horizontal_flip=True, 
                                   rotation_range=20, zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)

# Creer generateurs de donnees
train_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size=(256, 256), 
    batch_size=32, class_mode='binary')
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(256, 256), 
    batch_size=32, class_mode='binary')

# Construir le modele
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# entrainer le modele
history = model.fit(train_generator, 
                    epochs=20, 
                    validation_data=val_generator)

# Evaluer le modele
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# enregistrer le modele
model.save('fake_image_detector.h5')
