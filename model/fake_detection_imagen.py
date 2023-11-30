import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

np.random.seed(2)

sns.set(style='white', context='notebook', palette='deep')

from pylab import *
from PIL import Image, ImageChops, ImageEnhance

import os

# Define el directorio que contiene las imágenes
directory = 'data/imagenes'

# Lista para almacenar las rutas completas de los archivos
file_paths = []

import json
import pandas as pd

# Ubicación del archivo JSON
json_file_path = 'data/imagenes_descargadas.json'

# Leer el archivo JSON
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Crear listas para almacenar los datos
file_paths = []
targets = []

# Recorrer los datos y recopilar la información
for item in data:
    file_name = item['nombre_archivo']
    target = item['2_way_label']

    # Asumiendo que los nombres de archivo en el JSON no contienen la ruta completa
    file_path = f'data/imagenes/{file_name}'

    # Agregar la información a las listas
    file_paths.append(file_path)
    targets.append(target)

# Crear un DataFrame con los datos
df = pd.DataFrame({
    'Filepath': file_paths,
    'Target': targets
})

# Mostrar el DataFrame
print(df)

# Si necesitas guardar el DataFrame en un archivo CSV
df.to_csv('image_file_paths_with_target.csv', index=False)

# Suponiendo que 'df' es tu DataFrame original
dataset = df[:25000]

# Muestra el DataFrame
print(dataset)

# Si necesitas guardar el DataFrame en un archivo CSV
#df.to_csv('image_file_paths.csv', index=False)



def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]


def convert_to_ela_image(path, quality):
    filename = path
    resaved_filename = filename.split('.')[0] + '.resaved.jpg'
    ELA_filename = filename.split('.')[0] + '.ela.png'

    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)

    ela_im = ImageChops.difference(im, resaved_im)

    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    return ela_im

#Image.open('data/news/imagen_10.jpg')

#convert_to_ela_image('data/news/imagen_10.jpg', 90)

#Image.open('data/Fake/imagen_6.jpg')

#convert_to_ela_image('data/Fake/imagen_6.jpg', 90)

#dataset = pd.read_csv('datasets/dataset.csv')

X = []
Y = []

for index, row in dataset.iterrows():
    X.append(array(convert_to_ela_image(row[0], 90).resize((128, 128))).flatten() / 255.0)
    Y.append(row[1])

X = np.array(X)
Y = to_categorical(Y, 2)

X = X.reshape(-1, 128, 128, 3)


X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=5)

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'valid',
                 activation ='relu', input_shape = (128,128,3)))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'valid',
                 activation ='relu'))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))
print("Input: ", model.input_shape)
print("Output: ", model.output_shape)

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation = "softmax"))

model.summary()

#optimizer = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)
optimizer = RMSprop(learning_rate=0.0005, rho=0.9, epsilon=1e-08)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

#early_stopping = EarlyStopping(monitor='val_acc',
#                              min_delta=0,
#                              patience=2,
#                              verbose=0, mode='auto')

early_stopping = EarlyStopping(monitor='val_accuracy',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')

epochs = 30
batch_size = 100

history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,
          validation_data = (X_val, Y_val), verbose = 2, callbacks=[early_stopping])

# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation loss")
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred, axis=1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val, axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=range(2))

# Guardar el modelo en un archivo HDF5
model.save('modelo/mi_modelo_caso2.h5')  # Crea un archivo HDF5 'mi_modelo.h5'
