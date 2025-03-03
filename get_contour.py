import os
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2

test = '/path/to/DIS5K/DIS-TR/gt/'
save_path ='/path/to/DIS5K/DIS-TR/contour/'
os.makedirs(save_path)

to_test = {'contour':test}
img_transform = transforms.Compose([
    transforms.ToTensor()])
to_pil = transforms.ToPILImage()


for name, root in to_test.items():
    path = os.path.join(root)
    img_list = [os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith('.png')]
    for idx, img_name in enumerate(img_list):
        print('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
        img1 = Image.open(os.path.join(root, img_name + '.png')).convert('L')
        img1 = np.array(img1)
        kernel = np.ones((5,5),np.uint8)
        img2 = cv2.erode(img1,kernel)
        img3 = cv2.dilate(img1,kernel)
        img = np.array(img3-img2)
        img[img >= 6] = 255
        img[img<6] = 0
        cv2.imwrite(os.path.join(save_path, img_name + '.png'), img,[int(cv2.IMWRITE_JPEG_LUMA_QUALITY),50])