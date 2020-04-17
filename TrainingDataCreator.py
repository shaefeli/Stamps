import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import os
import matplotlib.image as mpimg
from TrainingDataParams import *
import shutil


class TrainingDataCreator:

    def __init__(self, image_size, possible_values, augmentation_actions):
        """
        :param possible_values: possible labels of the stamps
        :param augmentation_actions: a dictionary specifying which data augmentation is performed. Possible values are:
            -nr_uniform_background (int)
            -create_rectangular_images (bool, (max_added_width,max_added_height)) #Should we create rectangular images and resize them
            -helvetica_proba (float)
            -blur_proba (float)
            -cutout_proba (float)
        """
        self.image_size = image_size
        self.possibleValues = possible_values
        self.augmentation_actions = augmentation_actions
        self.number_of_created_images = self.calculate_number_of_images()
        np.random.seed(0)
        random.seed(0)
        print(self.number_of_created_images, "Training images can be created...")

    def calculate_number_of_images(self):
        nr_texture_images = 15
        nr_uniform_backgrounds = self.augmentation_actions["nr_uniform_background"]
        nr_base_images = nr_texture_images + nr_uniform_backgrounds
        if self.augmentation_actions["create_rectangular_images"][0]:
            nr_base_images *= 2

        nr_fonts = nr_chosen_fonts
        nr_size = 2
        nr_positions = nr_chosen_positions
        nr_possible_values = len(self.possibleValues)

        return nr_possible_values * nr_base_images * nr_fonts * nr_size * nr_positions

    def display_image(self, image):
        plt.imshow(image)
        plt.show()

    def create_training_data(self, path_to_output="", create_folder=True):
        background_images_square = []
        background_images_rect = []

        ######First create background images#####
        background_images_uniform_square = self.create_uniform_background(self.image_size[0], self.image_size[1])
        background_images_texture_square = self.create_image_background(self.image_size[0], self.image_size[1],
                                                                        resize=True)

        if self.augmentation_actions["create_rectangular_images"][0]:
            max_added_width = self.augmentation_actions["create_rectangular_images"][1][0]
            max_added_height = self.augmentation_actions["create_rectangular_images"][1][1]
            background_images_uniform_rect = self.create_uniform_background(self.image_size[0], self.image_size[1],
                                                                            max_added_width, max_added_height)
            background_images_texture_rect = self.create_image_background(self.image_size[0], self.image_size[1],
                                                                          resize=False)

        background_images_square.extend(background_images_uniform_square)
        background_images_square.extend(background_images_texture_square)
        background_images_rect.extend(background_images_uniform_rect)
        background_images_rect.extend(background_images_texture_rect)

        training_images_square = self.write_on_images(background_images_square, self.possibleValues,
                                                      self.augmentation_actions["helvetica_proba"])

        training_images_rect = self.write_on_images(background_images_rect, self.possibleValues,
                                                    self.augmentation_actions["helvetica_proba"])

        #######COMPLETE CODE FROM HERE######
        for label in training_images_square:
            training_images_square[label] = self.blur_images(training_images_square[label],
                                                             self.augmentation_actions["blur_proba"])
            training_images_rect[label] = self.blur_images(training_images_rect[label],
                                                           self.augmentation_actions["blur_proba"])

        for label in training_images_square:
            training_images_square[label] = self.random_cutout(training_images_square[label],
                                                               self.augmentation_actions["cutout_proba"])
            training_images_rect[label] = self.random_cutout(training_images_rect[label],
                                                             self.augmentation_actions["cutout_proba"])

        training_images = training_images_square
        for label in training_images_rect:
            training_images[label].extend(
                self.resize_rect(training_images_rect[label], self.image_size[0], self.image_size[1]))

        self.writeToDisk(training_images)

    def writeToDisk(self, images):
        for label in images:
            images_for_one_label = images[label]
            path_to_folder = os.path.join("training_images", label)
            if os.path.exists(path_to_folder) and os.path.isdir(path_to_folder):
                shutil.rmtree(path_to_folder)
            os.mkdir(path_to_folder)
            for i, img in enumerate(images_for_one_label):
                cv2.imwrite(os.path.join(path_to_folder, str(i + 1) + ".jpg"), img)

    def resize_rect(self, images, height, width):
        images_resized = []
        for img in images:
            dim = (width, height)
            resized = cv2.resize(np.int16(img), dim)
            images_resized.append(np.array(resized, dtype="int32"))
        return images_resized

    def random_cutout(self, images, cutout_proba):
        indices = np.random.choice(range(len(images)), int(cutout_proba * len(images)))
        cutout_images = []
        for j, image in enumerate(images):
            if j in indices:
                image = images[j].copy()
                im_height = image.shape[0]
                im_width = image.shape[1]
                # do 2 to 10 cutouts
                nr_cutouts = random.randint(nr_cutouts_min, nr_cutouts_max)
                for i in range(nr_cutouts):
                    # Get a random size between 5 and 10 pixels
                    size_cutout = random.randint(size_cutout_min, size_cutout_max)
                    # Get a random position (make sure that it stays within the boundaries)
                    top_x = random.randint(0, im_width - size_cutout - 2)
                    top_y = random.randint(0, im_height - size_cutout - 2)
                    image[top_y:top_y + size_cutout, top_x:top_x + size_cutout] = np.tile([0, 0, 0],
                                                                                          (size_cutout, size_cutout, 1))
            cutout_images.append(image)
        return cutout_images

    def blur_images(self, images, blur_proba):
        indices = np.random.choice(range(len(images)), int(blur_proba * len(images)))
        blured_images = []
        for j, image in enumerate(images):
            if j in indices:
                image = images[j]
                blured_images.append(np.int32(cv2.GaussianBlur(np.float32(image), (size_kernel_blur, size_kernel_blur), 1)))
            else:
                blured_images.append(image)
        return blured_images

    def write_on_images(self, images, labels, helvetica_probability):
        all_images = dict()
        for label in labels:
            images_per_label = []
            for image in images:
                im_height = image.shape[0]
                im_width = image.shape[1]
                fonts = np.random.choice(range(8), nr_chosen_fonts)
                positions = np.random.choice(range(8), nr_chosen_positions)
                for font in fonts:
                    for position in positions:
                        # Either a corner, or a center of a side, starts from corner top left, clockwise
                        for size in range(2):
                            # Either big or small (0 for big, 1 for small)

                            font_scale, thickness = self.calculate_font_scale_and_thickness(label, im_width, im_height,
                                                                                            size, font, ratio_big_font,
                                                                                            ratio_small_font)

                            position_coord = self.calculate_label_position(label, im_width, im_height, font, position,
                                                                           font_scale, thickness)

                            color = random_colors[random.randint(0, nr_random_colors - 1)]
                            new_image = self.write_on_image(image.copy(), label, position_coord, font, font_scale,
                                                            thickness, color)

                            if (np.random.binomial(1, helvetica_probability) == 1):
                                position_helvetica, font_scale_helvetica, thickness = self.compute_helvetica_placement(
                                    im_height, im_width, font, size, position)

                                new_image = self.write_on_image(new_image, helvetica_label, position_helvetica, font,
                                                                font_scale_helvetica, thickness, color)

                            images_per_label.append(new_image)

            all_images[label] = images_per_label
        return all_images

    def compute_helvetica_placement(self, im_height, im_width, font, size, position):
        label = helvetica_label

        font_scale, thickness = self.calculate_font_scale_and_thickness(label, im_width, im_height, size, font,
                                                                        ratio_big_font_helvetica,
                                                                        ratio_small_font_helvetica)

        size_x, size_y = cv2.getTextSize(label, font, font_scale, thickness)[0]

        margin_x = int(margin_ratio * size_x)
        margin_y = int(margin_ratio * size_y)

        if margin_x <= 1:
            margin_x = 2
        if margin_y <= 1:
            margin_y = 2

        pos_x = 0
        pos_y = 0

        if position == 0 or position == 1 or position == 2 or position == 3:  # Put it at the bottom left
            pos_x = margin_x
            pos_y = im_height - margin_y

        elif position == 4 or position == 5 or position == 6 or position == 7:
            pos_x = im_width - size_x - margin_x
            pos_y = size_y + margin_y

        return (pos_x, pos_y), font_scale, thickness

    def calculate_font_scale_and_thickness(self, label, im_width, im_height, size, font, ratio_big_font,
                                           ratio_small_font):
        thickness = thickness_small

        if size == big_text_value:
            thickness == thickness_big

        if label == helvetica_label:
            thickness = thickness_helvetica

        size_x, size_y = cv2.getTextSize(label, font, 1, thickness)[0]
        ratio_y, ratio_x = size_y / im_height, size_x / im_width
        font_scale = 1
        if ratio_x > ratio_y:  # Do according to ratio_x
            if size == 0:
                font_scale = ratio_big_font * im_width / size_x
            else:
                font_scale = ratio_small_font * im_width / size_x

        else:  # Do according to ratio y
            if size == 0:
                font_scale = ratio_big_font * im_height / size_y
            else:
                font_scale = ratio_small_font * im_height / size_y

        return font_scale, thickness

    def calculate_label_position(self, label, im_width, im_height, font, position, font_scale, thickness):
        size_x, size_y = cv2.getTextSize(label, font, font_scale, thickness)[0]

        margin_x = int(margin_ratio * size_x)
        margin_y = int(margin_ratio * size_y)

        if margin_x <= 1:
            margin_x = 2
        if margin_y <= 1:
            margin_y = 2

        if position == 0:
            pos_x = margin_x
            pos_y = size_y + margin_y
            return pos_x, pos_y

        elif position == 1:
            pos_x = int((im_width / 2)) - int((size_x / 2))
            pos_y = size_y + margin_y
            return pos_x, pos_y

        elif position == 2:
            pos_x = im_width - size_x - margin_x
            pos_y = size_y + margin_y
            return pos_x, pos_y

        elif position == 3:
            pos_x = im_width - size_x - margin_x
            pos_y = int((im_height / 2)) + int((size_y / 2))
            return pos_x, pos_y

        elif position == 4:
            pos_x = im_width - size_x - margin_x
            pos_y = im_height - margin_y
            return pos_x, pos_y

        elif position == 5:
            pos_x = int((im_width / 2)) - int((size_x / 2))
            pos_y = im_height - margin_y
            return pos_x, pos_y

        elif position == 6:
            pos_x = margin_x
            pos_y = im_height - margin_y
            return pos_x, pos_y

        elif position == 7:
            pos_x = margin_x
            pos_y = int((im_height / 2)) + int((size_y / 2))
            return pos_x, pos_y

    def write_on_image(self, image, label, position, font, fontScale, thickness, color):

        lineType = 1  # Keep this one constant set to 1
        cv2.putText(image, label, position, font, fontScale, color, thickness, lineType)

        return image

    def create_uniform_background(self, height, width, diff_height=0, diff_width=0):
        images = []
        for i in range(self.augmentation_actions["nr_uniform_background"]):
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if diff_height == 0 and diff_width == 0:
                uniform_image = np.tile(random_color, (height, width, 1))
                images.append(uniform_image)
            else:
                uniform_image = np.tile(random_color, (height + random.randint(0, diff_height),
                                                       width + random.randint(0, diff_width), 1))
                images.append(uniform_image)

        return images

    def create_image_background(self, height, width, resize=False, im_folder="background_images"):
        images = []
        for i in range(1, 16):
            filename = os.path.join(im_folder, str(i) + ".jpg")
            images.append(mpimg.imread(filename))

        if resize:
            images_resized = []
            for img in images:
                dim = (width, height)
                resized = cv2.resize(img, dim)
                images_resized.append(resized)
            return images_resized

        return images
