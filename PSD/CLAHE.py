import os
import cv2

# Code for generating dehaze images by CLAHE (in order to calculate Loss_CLAHE)
if __name__ == '__main__':
    data_path = '.'
    out_path = '/output/'
    name_list = list(os.walk(data_path))[0][2]
    for i, name in enumerate(name_list):
        img0 = cv2.imread(data_path + name)
        b,g,r = cv2.split(img0)
        img_rgb = cv2.merge([r,g,b])
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(16,16))
        img_rgb2 = clahe.apply(img_rgb.reshape(-1)).reshape(img_rgb.shape)
        r, g, b = cv2.split(img_rgb2)
        img_out = cv2.merge([b, g, r])
        cv2.imwrite(out_path)
