import glob
import cv2

dir = './new_maps/'

images = glob.glob(dir + '/*.png')

for img_path in images:
    image = cv2.imread(img_path, 0)
    image = (image - image.min()) / (image.max() - image.min()) * 255.
    image = 255. - image
    cv2.imwrite(img_path, image)