import cv2
import glob

images = glob.glob("*.jpg")
''' 
print(type(img))
print(img)
print(img.shape)
print(img.ndim) '''

for image in images:
    img = cv2.imread(image, 1)
    re = cv2.resize(img,(1000, 1000))
    cv2.imshow("2", re)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    cv2.imwrite("resized" + image,re)