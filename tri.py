import os
import shutil
import csv
import json

def organize_images_by_class(csv_file_path, image_folder_path, output_folder_path):
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_folder_path, exist_ok=True)

    # Lire le fichier CSV
    with open(csv_file_path, mode='r', newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            filename = row['filename']
            region_attributes = json.loads(row['region_attributes'])

            # Vérifier si l'image a une classe
            if 'object' in region_attributes:
                object_class = region_attributes['object']

                # Créer un dossier pour la classe s'il n'existe pas
                class_folder = os.path.join(output_folder_path, object_class)
                os.makedirs(class_folder, exist_ok=True)

                # Chemin de l'image source et destination
                src_path = os.path.join(image_folder_path, filename)
                dest_path = os.path.join(class_folder, filename)

                # Copier l'image dans le dossier de la classe
                shutil.copy(src_path, dest_path)
                print(f"Copié {filename} dans {class_folder}")
            else:
                class_folder = os.path.join(output_folder_path, "rien")
                os.makedirs(class_folder, exist_ok=True)

                # Chemin de l'image source et destination
                src_path = os.path.join(image_folder_path, filename)
                dest_path = os.path.join(class_folder, filename)

                # Copier l'image dans le dossier de la classe
                shutil.copy(src_path, dest_path)

                print(f"Copié {filename} dans 'rien'")

# Exemple d'utilisation
csv_file_path = 'annotations.csv'  # Chemin vers le fichier CSV
image_folder_path = 'images'       # Dossier contenant les images
output_folder_path = 'output'      # Dossier de sortie pour les images organisées

organize_images_by_class(csv_file_path, image_folder_path, output_folder_path)