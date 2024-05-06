#Structural Similarity Index (SSIM)
import cv2
import numpy as np
from skimage.metrics import structural_similarity

# Load images
first = cv2.imread('clownfish_1.jpg')
second = cv2.imread('clownfish_2.jpg')

# Convert images to grayscale
first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

# Compute SSIM between two images
score, diff = structural_similarity(first_gray, second_gray, full=True)
print("Similarity Score: {:.3f}%".format(score * 100))

# Convert the diff image to 8-bit unsigned integers
diff = (diff * 255).astype("uint8")

# Threshold the difference image
_, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Find contours to obtain the regions that differ between the two images
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Highlight differences
mask = np.zeros(first.shape, dtype='uint8')
filled = second.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area > 100:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(first, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.rectangle(second, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
        cv2.drawContours(filled, [c], 0, (0, 255, 0), -1)

# Display images
cv2.imshow('first', first)
cv2.imshow('second', second)
cv2.imshow('diff', diff)
cv2.imshow('mask', mask)
cv2.imshow('filled', filled)
cv2.waitKey(0)
cv2.destroyAllWindows()
