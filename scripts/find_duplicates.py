from imagededup.methods import PHash

if __name__ == '__main__':
    phasher = PHash()

    # Generate encodings for all images in an image directory
    encodings = phasher.encode_images(image_dir='C:/Users/admin/Development/cyclist-ai-training/datasets/cyclist_dataset/images/doubt')

    # Find duplicates using the generated encodings
    duplicates = phasher.find_duplicates(encoding_map=encodings)

    # Print only keys with non-empty lists (i.e., where duplicates exist)
    for key, value in duplicates.items():
        if value:  # Check if the list is not empty
            print(f"Image: {key} has duplicates: {value}")