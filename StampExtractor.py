import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import cv2

#####################################
########HELPER METHODS###############
#####################################

def applyCanny(color_img):
    edges = cv2.Canny(color_img, 150, 200)
    return edges


def dbscan(coord):
    clustering = DBSCAN(eps=2, min_samples=3).fit(coord)
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    clusters = []
    thresh_nr_samples_for_cluster = 500
    for label in unique_labels:
        if label != -1:
            samples = np.where(labels == label)
            if len(samples[0]) > thresh_nr_samples_for_cluster:
                clusters.append(samples)
    return clusters


def image_to_coordinates(edgeImages):
    coord1, coord2 = np.where(edgeImages == 255)
    return np.array(list(zip(coord1, coord2)))


def filter_substamps(stamps):
    filtered_stamps = []
    for stamp in stamps:
        stamp_is_not_sub = True
        for other_stamp in stamps:
            if stamp[0] > other_stamp[0] and stamp[1] < other_stamp[1] and stamp[2] > other_stamp[2] \
                    and stamp[3] < other_stamp[3]:
                stamp_is_not_sub = False
                break
        if stamp_is_not_sub:
            filtered_stamps.append(stamp)
    return filtered_stamps


def extract_stamps_from_clusters(img, clusters, coordinates):
    # First gather all the coordinates of all the corners
    margin = 10
    img_height = len(img)
    img_width = len(img[0])
    stamp_corners = []
    for cluster in clusters:
        coordinates_cluster = coordinates[cluster]
        x_coord = np.array([x[0] for x in coordinates_cluster])
        y_coord = np.array([x[1] for x in coordinates_cluster])
        max_x, min_x = np.amax(x_coord), np.amin(x_coord)
        max_y, min_y = np.amax(y_coord), np.amin(y_coord)

        if min_x - margin >= 0:
            min_x = min_x - margin
        if min_y - margin >= 0:
            min_y = min_y - margin
        if max_x + margin < img_width:
            max_x = max_x + margin
        if max_y + margin < img_height:
            max_y = max_y + margin
        stamp_corners.append((min_x, max_x, min_y, max_y))

    # Filter the ones that are included in another stamp (the substamps, due to dbscan errors)
    filtered_corners = filter_substamps(stamp_corners)
    stamps = []
    for stamp in filtered_corners:
        img_stamp = img[stamp[0]:stamp[1], stamp[2]:stamp[3]]
        stamps.append(img_stamp)
    return stamps


###############################
#######MAIN FUNCTIONS##########
###############################

def extract_stamps_with_intermediate_results(original_image):
    """
    Function to extract the stamps from the image. Also returns the intermediate results
     ...
    :param original_image: the image from a page of stamps
    :returns: stamps: a list of images with all stamps\n
    canny_image: the canny image extracted from original_image\n
    coordinates: the coordinates of the points detected by canny\n
    clusters: the cluster assignment from dbscan (list of list of indices from coordinates)
    """
    canny_image = applyCanny(original_image)
    coordinates = image_to_coordinates(canny_image)
    clusters = dbscan(coordinates)
    stamps = extract_stamps_from_clusters(original_image, clusters, coordinates)
    return stamps, canny_image, coordinates, clusters


def extract_stamps(original_image):
    """
    Same function than extract_stamps_with_intermediate_results but only returns stamps
    ...
    :param original_image: the image from a page of stamps
    :return:
    stamps: a list of images with all stamps
    """
    stamps, _, _, _ = extract_stamps_with_intermediate_results(original_image)
    return stamps


################################
#######DISPLAY FUNCTIONS########
################################

def show_dbscan_clusters(img, coordinates, clusters):
    """
    Function to display the clusters found by dbscan
    ...
    :param img: the image we should draw on (a copy is made)
    :param coordinates: the coordinates of the points detected by canny
    :param clusters: the clusters returned by dbscan
    :return: nothing, displays the image with the clusters
    """
    img = img.copy()
    for cluster in clusters:
        coordinates_cluster = coordinates[cluster]
        random_color = np.random.randint(0, 255, 3)
        for i, j in coordinates_cluster:
            img[i:i + 3, j - 3:j + 3] = random_color
    plt.imshow(img)
    plt.show()


def show_canny(canny_image):
    """
    Function to display the canny image
    ...
    :param canny_image: The canny image to be displayed
    :return: nothing, but displays the image
    """
    plt.imshow(canny_image)
    plt.show()


def display_stamps(stamps):
    """
    Function that displays the stamps in a single image
    ...
    :param stamps: The found stamps
    :return: Nothing, but displays the stamps in one nice image
    """
    columns = 5
    rows = len(stamps) / 4 + 1
    fig = plt.figure(figsize=(8, 8))
    for i, stamp in enumerate(stamps):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(stamp)
    plt.show(block=True)
