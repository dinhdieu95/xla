# Dense Vector Representations
# tỉ lệ khớp chuẩn hơn 
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob

# Load the OpenAI CLIP Model
print('Loading CLIP Model...')
model = SentenceTransformer('clip-ViT-B-32')

# Load image names
image_names = list(glob.glob('./*.jpg'))
print("Images:", len(image_names))

# Encode images
print("Encoding images...")
encoded_images = []
for filepath in image_names:
    encoded_image = model.encode(Image.open(filepath))
    encoded_images.append(encoded_image)

# Now we run the clustering algorithm. This function compares images against 
# all other images and returns a list with the pairs that have the highest 
# cosine similarity score
print("Finding similar images...")
processed_images = util.paraphrase_mining_embeddings(encoded_images)
NUM_SIMILAR_IMAGES = 10 

# =================
# DUPLICATES
# =================
print('Finding duplicate images...')
# Filter list for duplicates. Results are triplets (score, image_id1, image_id2) and are sorted in decreasing order
# A duplicate image will have a score of 1.00
# It may be 0.9999 due to lossy image compression (.jpg)
duplicates = [image for image in processed_images if image[0] >= 0.999]

# Output the top X duplicate images
for score, image_id1, image_id2 in duplicates[0:NUM_SIMILAR_IMAGES]:
    print("\nScore: {:.3f}%".format(score * 100))
    print(image_names[image_id1])
    print(image_names[image_id2])

# =================
# NEAR DUPLICATES
# =================
print('Finding near duplicate images...')
# Use a threshold parameter to identify two images as similar. By setting the threshold lower, 
# you will get larger clusters which have less similar images in it. Threshold 0 - 1.00
# A threshold of 1.00 means the two images are exactly the same. Since we are finding near 
# duplicate images, we can set it at 0.99 or any number 0 < X < 1.00.
threshold = 0.99
near_duplicates = [image for image in processed_images if image[0] < threshold]

for score, image_id1, image_id2 in near_duplicates[0:NUM_SIMILAR_IMAGES]:
    print("\nScore: {:.3f}%".format(score * 100))
    print(image_names[image_id1])
    print(image_names[image_id2])


# Set a distance threshold to filter out matches
DISTANCE_THRESHOLD = 50

# # Display the top unmatching images for each image
# for i, matches in enumerate(matches_list):
#     print(f"Top unmatching images for {image_paths[i]}:")
#     for j, image_match in enumerate(image_paths):
#         if j != i:  # Skip self-matches
#             # Check if there is a match with distance above the threshold
#             is_matching = all(match.distance >= DISTANCE_THRESHOLD for match in matches)
#             if not is_matching:
#                 print(f"    - {image_match}")