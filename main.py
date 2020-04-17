from ImageHandler import *
from StampExtractor import *
from TrainingDataCreator import *
from StampClassifier_from_synthetic import *

# Stamp extraction part works well with classical computer vision
# Synthetic data not close enough to real stamps to work.
# If more data, could to transfer learning from svnh datataset trained model


create_training_data = False
extract_and_write_stamps = False
train_model = False
compute_total_sum = True

if extract_and_write_stamps:
    imageFolder = "images"
    imHandler = ImageHandler(imageFolder)
    originalImage = imHandler.get_image_from_name("image1.jpg")
    stamps, canny, coordinates, clusters = extract_stamps_with_intermediate_results(originalImage)
    # imHandler.display_image(originalImage)
    # show_canny(canny)
    # show_dbscan_clusters(originalImage, coordinates, clusters)
    # display_stamps(stamps)
    imHandler.writeToDisk(stamps, "stamp_images")

if create_training_data:
    training_data_creator = TrainingDataCreator([100, 100],
                                                ["5", "10", "15", "20", "25", "30", "50", "75", "80", "150", "250"],
                                                {"nr_uniform_background": 2,
                                                 "create_rectangular_images": (True, (50, 50)),
                                                 "helvetica_proba": 0.8, "blur_proba": 0.05,
                                                 "cutout_proba": 0.08})
    training_data_creator.create_training_data()

sc = StampClassifierSynthetic()
if train_model:
    sc.train()

if compute_total_sum:
    a, b = sc.predict("stamp_images")
    predictions = [(a[i],b[i]) for i in range(len(a))]
    print("The value of the page is:", np.sum(np.int32(a)) * 0.01, "CHF")
