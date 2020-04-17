# Stamps

Project that takes as input an image file, containing multiple stamps. The final goal is to output the total value of the stamp page.

It first uses classical computer vision to cut out the individual stamps from the page: it first detects the edges of the image using the Canny Edge Detection, and detects the location of the stamps by first proceding to a DBSCAN clustering, and then using the outmost coordinates of the cluster. 
In a second part (not quite working part), 25'000 synthetic training images are created using OpenCV representing individual stamps. A simple Keras CNN model is then trained onto these images. Classification can then be performed. 
The second part is not working because the distribution of the synthetic images is too far away from the one of real stamps. However, there is a lack of real stamps for training images (only 2 pages in total, with approximately 30 stamps each). With more training images (~1000 stamps maybe), we could use transfer learning from a model pretrained on the well-known Google's SVNH dataset (steet view house numbers).  
