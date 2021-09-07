from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

# Function to show array of images (intermediate results)
def show_images(images):
	for i, img in enumerate(images):
		cv2.imshow("image_" + str(i), img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

img_path = r"Please provide image path here"

# Read image and preprocess
image = cv2.imread(img_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 2)
kernal =     kernel = np.ones((5, 5), np.uint8)
morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

ht, wd, _ = image.shape
ref_object = thresh[0:int(ht/10), 0:int(wd/10)]
# show_images([blur, thresh, ref_object, morphed])

# Find contours
refcnts = cv2.findContours(ref_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
refcnts = imutils.grab_contours(refcnts)

cnts = cv2.findContours(morphed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Sort contours from left to right as leftmost contour is reference object
(cnts, _) = contours.sort_contours(cnts)
(refcnts, _) = contours.sort_contours(refcnts)

# Remove contours which are not large enough
cnts = [x for x in cnts if cv2.contourArea(x) > 5000]
refcnts = [x for x in refcnts if cv2.contourArea(x) > 100]

#cv2.drawContours(image, cnts, -1, (0,255,0), 3)

#show_images([image, edged])
#print(len(cnts))

# Reference object dimensions
# Here for reference I have used a 2cm x 2cm square
ref_object = refcnts[0]
box = cv2.minAreaRect(ref_object)
box = cv2.boxPoints(box)
box = np.array(box, dtype="int")
box = perspective.order_points(box)
(tl, tr, br, bl) = box
dist_in_pixel = euclidean(tl, tr)
dist_in_cm = 2
pixel_per_cm = dist_in_pixel/dist_in_cm

# Draw remaining contours
cnt = cnts[0]
box = cv2.minAreaRect(cnt)
box = cv2.boxPoints(box)
box = np.array(box, dtype="int")
box = perspective.order_points(box)
(tl, tr, br, bl) = box
cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2),
                     tl[1] + int(abs(tr[1] - tl[1])/2))
mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2),
                   tr[1] + int(abs(tr[1] - br[1])/2))
wid = euclidean(tl, tr)/pixel_per_cm
ht = euclidean(tr, br)/pixel_per_cm
cv2.putText(image, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
cv2.putText(image, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
print(" Height of plant: {} cm, width of plant:{} cm".format(ht, wid))
show_images([image])
