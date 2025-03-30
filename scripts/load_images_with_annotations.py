import PIL
from PIL import Image
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Lecture du CSV contenant les informations relatives a la base de donnees
dataset = []
with open('cyclist-ai-training/annotations_dispatched.csv', newline='') as csvfile:
	filereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in filereader:
		data = row[0].split(',')
		if data[0] != 'filename':
			xmin = float(data[2])
			ymin = float(data[3])
			xmax = float(data[4])
			ymax = float(data[5])
			size = float(256)
			box = [xmin, ymin, xmax, ymax]
			new_entry = {'type': data[1],'class': data[6], 'path': data[0], 'shape': [size, size], 'box': box}
			dataset.append(new_entry)

# Nombre de classes de la base de donnees et intitule des classes
class_labels = list(dict.fromkeys([item['class'] for item in dataset]))
num_classes = len(class_labels)

dataset_train = [item for item in dataset if item['type']=='train']
dataset_test = [item for item in dataset if item['type']=='test']
dataset_val = [item for item in dataset if item['type']=='val']

def build_localization_tensors(image_size, dataset, num_classes):
  x = np.zeros((len(dataset), image_size, image_size, 3))
  y = np.empty((len(dataset), num_classes + 5))

  i = 0

  for item in dataset:
    img = Image.open('cyclist-ai-training/images/' + item['path'])
    img = img.resize((image_size,image_size), Image.Resampling.LANCZOS)
    img = img.convert('RGB')
    x[i] = np.asarray(img)

    y[i, 0] = 1

    # Coordonnées de boîte englobante
    img_shape = item['shape']
    box = item['box']
    bx = (box[0] + (box[2] - box[0])/2)/img_shape[0]
    by = (box[1] + (box[3] - box[1])/2)/img_shape[1]
    bw = (box[2] - box[0])/img_shape[0]
    bh = (box[3] - box[1])/img_shape[1]
    y[i, 1] = bx
    y[i, 2] = by
    y[i, 3] = bw
    y[i, 4] = bh

    # Probabilités de classe, sous la forme d'une one-hot vector
    label = class_labels.index(item['class'])
    classes_probabilities = keras.utils.to_categorical(label, num_classes=num_classes)
    y[i, 5:] = classes_probabilities

    i = i+1

  return x, y

IMAGE_SIZE = 128

# Lecture des données d'entraînement, de tests et de validations
x_train, y_train = build_localization_tensors(IMAGE_SIZE, dataset_train, num_classes)
x_test, y_test = build_localization_tensors(IMAGE_SIZE, dataset_test, num_classes)
x_val, y_val = build_localization_tensors(IMAGE_SIZE, dataset_val, num_classes)

# Pour améliorer l'entraînement, on peut centrer-réduire les coordonnées des bounding boxes...
y_std = np.std(y_train, axis=0)
y_mean = np.mean(y_train, axis=0)
y_train[...,1:5] = (y_train[...,1:5] - y_mean[1:5])/y_std[1:5]
y_val[...,1:5] = (y_val[...,1:5] - y_mean[1:5])/y_std[1:5]
y_test[...,1:5] = (y_test[...,1:5] - y_mean[1:5])/y_std[1:5]

x_train = x_train/255
x_val = x_val/255
x_test = x_test/255


def print_data_localisation(x, y, y_pred=[], id=None, image_size=64):
  if id==None:
    # Tirage aléatoire d'une image dans la base
    num_img = np.random.randint(x.shape[0]-1)
  else:
    num_img = id

  img = x[num_img]
  lab = y[num_img]

  colors = ["blue", "red", "green", "yellow"] # Différentes couleurs pour les différentes classes
  classes = ['velo', 'velo_personne', 'velo_personne_casque', 'rien']

  if np.any(y_pred):
    plt.subplot(1, 2, 1)

  plt.imshow(img)
  # Détermination de la classe
  class_id = np.argmax(lab[5:])

  # Détermination des coordonnées de la boîte englobante dans le repère image
  ax = (lab[1]*y_std[1] + y_mean[1]) * image_size
  ay = (lab[2]*y_std[2] + y_mean[2]) * image_size
  width = (lab[3]*y_std[3] + y_mean[3]) * image_size
  height = (lab[4]*y_std[4] + y_mean[4]) * image_size

  # Détermination des extrema de la boîte englobante
  p_x = [ax-width/2, ax+width/2]
  p_y = [ay-height/2, ay+height/2]
  # Affichage de la boîte englobante, dans la bonne couleur
  plt.plot([p_x[0], p_x[0]],p_y,color=colors[class_id])
  plt.plot([p_x[1], p_x[1]],p_y,color=colors[class_id])
  plt.plot(p_x,[p_y[0],p_y[0]],color=colors[class_id])
  plt.plot(p_x,[p_y[1],p_y[1]],color=colors[class_id])
  plt.title("Vérité Terrain : Image {} - {}".format(num_img, classes[class_id]))

  plt.show()

for i in range(10):
    print_data_localisation(x_train, y_train, image_size=IMAGE_SIZE, id=i)
for i in range(10):
    print_data_localisation(x_val, y_val, image_size=IMAGE_SIZE, id=i)
for i in range(10):
    print_data_localisation(x_test, y_test, image_size=IMAGE_SIZE, id=i)