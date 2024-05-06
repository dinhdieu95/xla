import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# Load images
# before = cv2.imread('left.jpg')
# after = cv2.imread('right.jpg')
before = cv2.imread('img1.jpg')
after = cv2.imread('img2.jpg')
print("Image 1 shape:", before.shape)
print("Image 2 shape:", after.shape)
# (853, 614)

before = cv2.resize(before, (853, 614))
after = cv2.resize(after, (853, 614))
# Convert images to grayscale
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Compute SSIM between the two images
(score, diff) = ssim(before_gray, after_gray, full=True)
print("Image Similarity: {:.4f}%".format(score * 100))

# Convert the difference image to 8-bit unsigned integers
diff = (diff * 255).astype("uint8")
diff_box = cv2.merge([diff, diff, diff])

# Threshold the difference image
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Find contours to obtain the regions of the two input images that differ
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area > 40:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
        cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

# Display images
cv2.imshow('before', before)
cv2.imshow('after', after)
cv2.imshow('diff', diff)
cv2.imshow('diff_box', diff_box)
cv2.imshow('mask', mask)
cv2.imshow('filled after', filled_after)
cv2.waitKey(0)
