import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import shutil
import cv2
import numpy as np


class ImageHandler:
    def __init__(self, path_to_images):
        self.imageFolder = path_to_images
        valid, valid_images, faulty_names = self.verify_image_names(path_to_images)
        self.imagePaths = valid_images
        if not valid:
            print("Unvalid image names in the given folder:", path_to_images)
            print("Correct them before using more functions")
            for faulty_name in faulty_names:
                print(faulty_name)

    def verify_image_names(self, folderPath):
        faulty_names = []
        image_file_names = []
        valid = True
        expected_filename = r"image\d+.jpg"
        for filename in os.listdir(self.imageFolder):
            if not re.search(expected_filename, filename):
                faulty_names.append(filename)
                valid = False
            else:
                image_file_names.append(os.path.join(folderPath, filename))

        return valid, image_file_names, faulty_names

    def display_image(self, image):
        plt.imshow(image)
        plt.show()

    def get_image_from_name(self, filename):
        filename = os.path.join(self.imageFolder, filename)
        if filename in self.imagePaths:
            img = mpimg.imread(filename)
            return img
        else:
            print("Image doesn't exist!")

    def writeToDisk(self, images, folderName):
        if os.path.exists(folderName) and os.path.isdir(folderName):
            shutil.rmtree(folderName)
        os.mkdir(folderName)
        for i, img in enumerate(images):
            cv2.imwrite(os.path.join(folderName, str(i + 1) + ".jpg"), np.int32(img))
