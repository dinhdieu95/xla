import numpy as np
import cv2
from matplotlib import pyplot as plt

# ): Để hiển thị hình ảnh (đã được chú thích trong đoạn mã).

# img1 is the queryImage
# img2 is the trainImage

#img1 = cv2.imread('01-IMG_0491.jpg',0)
#img1 = cv2.imread('02-IMG_0492.jpg',0)
#img1 = cv2.imread('03-IMG_0493.jpg',0)
#img1 = cv2.imread('04-IMG_0494.jpg',0)
img1 = cv2.imread('06-IMG_0496.jpg',0)
#img1 = cv2.imread('07-IMG_0497.jpg',0)
#img1 = cv2.imread('08-IMG_0498.jpg',0)
#img1 = cv2.imread('10-IMG_0500.jpg',0)

# img2 = cv2.imread('heart_template.jpg',0)
img2 = cv2.imread('testcheo.jpg',0)
#img2 = cv2.imread('star_template.jpg',0)

# Initiate SIFT detector  - Khởi tạo bộ phát hiện SIFT
sift = cv2.SIFT_create() 

# find the keypoints and descriptors with SIFT , Phát hiện các điểm chính và tính toán các mô tả cho cả hai hình ảnh:
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters tham số đầu vào 
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

search_params = dict(checks=50)   # or pass empty dictionary --  hoặc truyền vào từ điển trống

flann = cv2.FlannBasedMatcher(index_params,search_params) # --Tạo một bộ khớp FLANN

matches = flann.knnMatch(des1,des2,k=2)  #Sử dụng bộ khớp FLANN để tìm k hàng xóm gần nhất cho mỗi mô tả trong hình ảnh truy vấn (des1) từ hình ảnh mẫu (des2)

# Need to draw only good matches, so create a mask --
matchesMask = [[0,0] for i in range(len(matches))]
good = []
for m, n in matches:
    # only preserve the matches are close enough to each other
    if m.distance < 0.7*n.distance:
        good.append(m)


# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):  # enumerate sử dụng để lặp qua một dãy các phần tử trong một cấu trúc
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]


draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

# errors if the line below is not commented out
# img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None,matchesThickness=2, **draw_params)
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None,matchesThickness=2, **draw_params)

# img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, **draw_params)

# draw matches on the image
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, matchesThickness=2,  flags=2)

# display the result
plt.imshow(img3, 'gray'), plt.show()


# what is the best way to quantify the how strong the match is?
# Đếm số lượng khớp 
count_matches = 0
for i in range(len(matches)):
	if matchesMask[i] == [1,0]:
		count_matches += 1
print (count_matches)
print("The number of element in image 1:", len(matches))
print("The number of element in image 2:", len(matchesMask))
plt.imshow(img3,),plt.show()
cv2.imwrite('abc.jpg', img3)