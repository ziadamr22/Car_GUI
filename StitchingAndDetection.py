import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh):
    keypoints_train_img = np.float32([keypoint.pt for keypoint in keypoints_train_img])
    keypoints_query_img = np.float32([keypoint.pt for keypoint in keypoints_query_img])

    if len(matches) >4:
        points_train = np.float32([keypoints_train_img[m.queryIdx] for m in matches])
        points_query = np.float32([keypoints_query_img[m.trainIdx] for m in matches])
        (H, status) = cv.findHomography(points_train, points_query, cv.RANSAC, reprojThresh)
        return (matches, H, status)
    else:
        print("fail")
        return None
def _get_kp_features(image):

    (keypoints, features) = cv.SIFT_create().detectAndCompute(image, None)

    return (keypoints, features)
def create_matching_object(method, crossCheck):
    if method == 'sift' or method == 'surf':
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=crossCheck)
    return bf
def key_points_matching(features_train_img, features_query_img, method):
    bf = create_matching_object(method, crossCheck=True)
    best_matches = bf.match(features_train_img, features_query_img)
    rawMatches = sorted(best_matches, key=lambda x: x.distance)
    return rawMatches
def key_points_matching_KNN(features_train_img, features_query_img, ratio, method):
    bf = create_matching_object(method, crossCheck=False)
    rawMatches = bf.knnMatch(features_train_img, features_query_img, k=2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []
    for m,n in rawMatches:
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches
def maxmatch(train_photo,query_photo):
    kp1,fea1=_get_kp_features(train_photo, 'sift')
    kp2,fea2=_get_kp_features(query_photo, 'sift')
    matches=key_points_matching(fea1,fea2,method=feature_extraction_algo)
    return matches

def _adjust(StitchedImage):
    gray = cv.cvtColor(StitchedImage, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv.boundingRect(cnt)
    StitchedImage = StitchedImage[y:y + h, x:x + w]
    return StitchedImage

def _stitcher(query_photo, train_photo):
    query_photo_gray = cv.cvtColor(query_photo, cv.COLOR_RGB2GRAY)
    train_photo_gray = cv.cvtColor(train_photo, cv.COLOR_RGB2GRAY)
    keypoints_train_img, features_train_img = _get_kp_features(train_photo_gray)
    keypoints_query_img, features_query_img = _get_kp_features(query_photo_gray)
    if feature_to_match == 'bf':
        matches = key_points_matching(features_train_img, features_query_img, method=feature_extraction_algo)

    elif feature_to_match == 'knn':
        matches = key_points_matching_KNN(features_train_img, features_query_img, ratio=0.75,
                                          method=feature_extraction_algo)
    M = homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh=54)
    if M is None:
        print("Error!")
        return
    (matches, Homography_Matrix, status) = M
    width = query_photo.shape[1] + train_photo.shape[1]
    height = max(query_photo.shape[0], train_photo.shape[0])
    result = cv.warpPerspective(train_photo, Homography_Matrix, (width, height))
    result[0:query_photo.shape[0], 0:query_photo.shape[1]] = query_photo


    mapped_features_image = cv.drawMatches(train_photo, keypoints_train_img, query_photo, keypoints_query_img,
                                            matches[:100],
                                            None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(mapped_features_image)
    plt.axis('off')
    plt.show()
    return result

if __name__ == "__main__":
    feature_extraction_algo = 'sift'
    feature_to_match = 'knn'
    image_paths = glob.glob('InputImages/*.jpg')
    index = -1
    images = []
    for image in image_paths:
        img = cv.imread(image)
        images.append(img)
    for i in range(0,len(images)):
        index=index+1
        maxmatches=0
        for j in range(0,len(images)):
           keypoints_i ,features_i=_get_kp_features(images[index])
           keypoints_j,features_j = _get_kp_features(images[j])
           matches = key_points_matching_KNN(features_i, features_j, ratio=0.75,method=feature_extraction_algo)
           if len(matches)>maxmatches and index!=j:
               maxmatches = len(matches)
               pair=(index,j)
        if len(images)==1:exit(0)
        StitchedImage = _stitcher(images[pair[1]], images[pair[0]])
        StitchedImage=_adjust(StitchedImage)
        if StitchedImage.shape[1]==(images[pair[0]].shape[1])or StitchedImage.shape[1]==(images[pair[1]].shape[1]):
                StitchedImage = _stitcher(images[pair[0]], images[pair[1]])
                StitchedImage = _adjust(StitchedImage)
        images.pop(pair[0])
        images.pop(pair[1]-1)
        index=index-1
        images.insert(0,StitchedImage)
        cv.imwrite("Stitched_Panorama.jpg", StitchedImage)
