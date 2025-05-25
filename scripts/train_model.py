import csv
import os
import shutil
from collections import defaultdict
import random
from collections import Counter
import yaml
from pathlib import Path
from ultralytics import YOLO

image_attributes = defaultdict(list)

# Read the CSV file
with open('../annotations_formatted.csv', newline='') as csvfile:
    filereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in filereader:
        data = row[0].split(',')
        if data[0] != 'filename':
            filename = data[0]
            xmin = float(data[1])
            ymin = float(data[2])
            width = float(data[3])
            height = float(data[4])
            xmax = xmin + width
            ymax = ymin + height
            attributes = [
                xmin,  # x
                ymin,  # y
                xmax,  # width
                ymax,  # height
                data[5]          # classe   #ENFAIT SEUL ELEMENT QUI NOUS INTERESSE
            ]
            image_attributes[filename].append(attributes)

# Number of classes in the dataset and their labels
class_labels = list(dict.fromkeys([attr[4] for attributes_list in image_attributes.values() for attr in attributes_list]))
num_classes = len(class_labels)

print("\nClass labels:", class_labels)

# Calculer le nombre total d'images
total_images = len(image_attributes)

# Calculer le nombre d'images pour chaque ensemble
train_size = int(0.8 * total_images)
val_size = test_size = int(0.10 * total_images)

# Créer des listes pour stocker les images de chaque ensemble
train_images = []
val_images = []
test_images = []

# Mélanger les images pour une distribution aléatoire
all_images = list(image_attributes.keys())
random.seed(23)
random.shuffle(all_images)

# Remplir les ensembles
train_images = all_images[:train_size]
val_images = all_images[train_size:train_size + val_size]
test_images = all_images[train_size + val_size:]

# Créer les ensembles de données
train_set = {img: image_attributes[img] for img in train_images}
val_set = {img: image_attributes[img] for img in val_images}
test_set = {img: image_attributes[img] for img in test_images}

# Afficher les résultats
print("Ensemble d'entraînement :", train_set)
print("Ensemble de validation :", val_set)
print("Ensemble de test :", test_set)


# Fonction pour compter les classes dans un ensemble de données
def count_classes(dataset):
    class_counts = Counter()
    for image, boxes in dataset.items():
        for box in boxes:
            class_counts[box[-1]] += 1
    return class_counts

# Compter les classes dans chaque ensemble
train_class_distribution = count_classes(train_set)
val_class_distribution = count_classes(val_set)
test_class_distribution = count_classes(test_set)

# Afficher les distributions
print("Distribution des classes dans l'ensemble d'entraînement :", train_class_distribution)
print("Distribution des classes dans l'ensemble de validation :", val_class_distribution)
print("Distribution des classes dans l'ensemble de test :", test_class_distribution)

# Fonction pour convertir un ensemble de données en liste de dictionnaires
def dataset_to_csv_format(dataset, dataset_name):
    csv_data = []
    for filename, boxes in dataset.items():
        for box in boxes:
            csv_data.append({
                'filename': filename,
                'dataset': dataset_name,
                'x': box[0],
                'y': box[1],
                'width': box[2],
                'height': box[3],
                'classe': box[4]
            })
    return csv_data

# Convertir chaque ensemble de données
train_csv_data = dataset_to_csv_format(train_set, 'train')
val_csv_data = dataset_to_csv_format(val_set, 'val')
test_csv_data = dataset_to_csv_format(test_set, 'test')

# Fusionner toutes les données
all_csv_data = train_csv_data + val_csv_data + test_csv_data

# Écrire dans un fichier CSV
csv_filename = '../annotations_dispatched.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['filename', 'dataset', 'x', 'y', 'width', 'height', 'classe'])
    writer.writeheader()
    writer.writerows(all_csv_data)

print(f"Données exportées avec succès dans {csv_filename}")

# Fonction pour calculer les proportions des classes dans un ensemble de données
def calculate_proportions(class_counts, total_images):
    return {cls: count / total_images for cls, count in class_counts.items()}

# Calculer les proportions pour chaque ensemble
train_proportions = calculate_proportions(train_class_distribution, len(train_images))
val_proportions = calculate_proportions(val_class_distribution, len(val_images))
test_proportions = calculate_proportions(test_class_distribution, len(test_images))

# Obtenir toutes les classes uniques
all_classes = set(train_class_distribution.keys()).union(val_class_distribution.keys()).union(test_class_distribution.keys())

# Lecture du CSV contenant les informations relatives à la base de données
dataset = []
with open('../annotations_dispatched.csv', newline='') as csvfile:
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

# Nombre de classes de la base de données et intitulé des classes
class_labels = list(dict.fromkeys([item['class'] for item in dataset]))
num_classes = len(class_labels)

dataset_train = [item for item in dataset if item['type']=='train']
dataset_test = [item for item in dataset if item['type']=='test']
dataset_val = [item for item in dataset if item['type']=='val']

print(class_labels)

"""Création des fichiers d'annotations au format yolo"""

os.makedirs("../annotations_yolo_format/", exist_ok=True)

# Conversion CSV qui respecte le format YOLO
datasetPerImage = {}
dataset = []
with open('../annotations_formatted.csv', newline='') as csvfile:
	filereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in filereader:
		data = row[0].split(',')
		if data[0] != 'filename':
			x = float(data[1])
			y = float(data[2])
			w = float(data[3])
			h = float(data[4])
			size = float(256)
			new_entry = {"filename": data[0], "x": float(data[1]), "y": float(data[2]), "w": float(data[3]), "h": float(data[4]), "class": data[5]}
			if new_entry["filename"] in datasetPerImage.keys():
				datasetPerImage[new_entry["filename"]].append(new_entry)
			else:
				datasetPerImage[new_entry["filename"]] = [new_entry]

def classToNumber(className):
	if (className == "velo"):
		return "0"
	elif (className == "velo_personne"):
		return "1"
	else:
		return "2"

def imageIsEmpty(entry):
	return entry[0]["class"] == "rien"

SIZE = 256
for path in datasetPerImage.keys():
	entry = datasetPerImage[path]
	if not imageIsEmpty(entry):
		with open('../annotations_yolo_format/' + path[:-4]+'.txt', 'w') as f:
			for bb in entry:
				values = classToNumber(bb["class"]) + " " + str((bb["x"] + bb["w"]/2)/SIZE) + " " + str((bb["y"] + bb["h"]/2)/SIZE) + " " + str(bb["w"]/SIZE) + " " + str(bb["h"]/SIZE) + "\n"
				f.write(values)
	#Si on décide de créer un fichier avec rien dedans pour les images sans objet à détecter
	else:
		open('../annotations_yolo_format/' + path[:-4]+'.txt', 'w').close()



"""Création des répertoires"""

os.makedirs("../datasets/", exist_ok=True)

os.makedirs("../datasets/cyclist_dataset/", exist_ok=True)

os.makedirs("../datasets/cyclist_dataset/images/", exist_ok=True)
os.makedirs("../datasets/cyclist_dataset/images/train", exist_ok=True)
os.makedirs("../datasets/cyclist_dataset/images/test", exist_ok=True)
os.makedirs("../datasets/cyclist_dataset/images/val", exist_ok=True)

os.makedirs("../datasets/cyclist_dataset/labels/", exist_ok=True)
os.makedirs("../datasets/cyclist_dataset/labels/train", exist_ok=True)
os.makedirs("../datasets/cyclist_dataset/labels/test", exist_ok=True)
os.makedirs("../datasets/cyclist_dataset/labels/val", exist_ok=True)

"""Répartion des données (images et annotations) dans les répertoires correspondant"""

for file in os.listdir("../images/"):

  if file in train_set.keys():
    shutil.copy("../images/" + file, "../datasets/cyclist_dataset/images/train")
    shutil.copy("../annotations_yolo_format/" + file.split(".")[0] + '.txt', "../datasets/cyclist_dataset/labels/train")

  elif file in test_set.keys():
    shutil.copy("../images/" + file, "../datasets/cyclist_dataset/images/test")
    shutil.copy("../annotations_yolo_format/" + file.split(".")[0] + '.txt', "../datasets/cyclist_dataset/labels/test")

  elif file in val_set.keys():
    shutil.copy("../images/" + file, "../datasets/cyclist_dataset/images/val")
    shutil.copy("../annotations_yolo_format/" + file.split(".")[0] + '.txt', "../datasets/cyclist_dataset/labels/val")

  else :
    print("error the image is not in any dataset !!")
    break

"""Modèle YOLO v8

Création du fichier de configuration
"""

data = {
    'path': '../datasets/cyclist_dataset',
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'names': {
        0: 'velo',
        1: 'velo_personne',
        2: 'velo_personne_casque'
    }
}

file_path = Path('data.yaml')
with file_path.open('w') as f:
    yaml.dump(data, f, default_flow_style=False)

print(f"File created at: {file_path.resolve()}")

# Entrainement du modèle avec transfer learning sans gel

model2 = YOLO('yolov8l.pt')
model2.train(data='data.yaml', epochs=800, imgsz=256, patience = 100)
metrics2 = model2.val(conf=0.7)
results2 = model2.predict(
    "../datasets/cyclist_dataset/images/val/",  # dossier des images
    conf=0.7,
    imgsz=256, # seuil de confiance -> d'après courbe F1 conf
    save=True,                      # pour sauvegarder les résultats
    save_txt=False,
    agnostic_nms=True #Pour qu'on ait pas plusieurs boites englobantes sur un même cycliste
)
