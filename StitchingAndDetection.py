import imutils 
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh):
    keypoints_train_img = np.float32([keypoint.pt for keypoint in keypoints_train_img])
    keypoints_query_img = np.float32([keypoint.pt for keypoint in keypoints_query_img])

    if len(matches) >4:
        points_train = np.float32([keypoints_train_img[m.queryIdx] for m in matches])
        points_query = np.float32([keypoints_query_img[m.trainIdx] for m in matches])
        (H, status) = cv2.findHomography(points_train, points_query, cv2.RANSAC, reprojThresh)
        return (matches, H, status)
    else:
        print("fail")
        return None
def _get_kp_features(image):

    (keypoints, features) = cv2.SIFT_create().detectAndCompute(image, None)

    return (keypoints, features)
def create_matching_object(method, crossCheck):
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
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
    gray = cv2.cvtColor(StitchedImage, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    StitchedImage = StitchedImage[y:y + h, x:x + w]
    return StitchedImage

def _stitcher(query_photo, train_photo):
    query_photo_gray = cv2.cvtColor(query_photo, cv2.COLOR_RGB2GRAY)
    train_photo_gray = cv2.cvtColor(train_photo, cv2.COLOR_RGB2GRAY)
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
    result = cv2.warpPerspective(train_photo, Homography_Matrix, (width, height))
    result[0:query_photo.shape[0], 0:query_photo.shape[1]] = query_photo


    mapped_features_image = cv2.drawMatches(train_photo, keypoints_train_img, query_photo, keypoints_query_img,
                                            matches[:100],
                                            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(mapped_features_image)
    plt.axis('off')
    plt.show()
    return result

if __name__ == "__main__":
    # StitchedImage= np.zeros([0,0])
    feature_extraction_algo = 'sift'
    feature_to_match = 'knn'
    image_paths = glob.glob('InputImages/*.jpg')
    index = -1
    images = []
    for image in image_paths:
        img = cv2.imread(image)
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
        if len(images)>1:
            StitchedImage = _stitcher(images[pair[1]], images[pair[0]])
            StitchedImage=_adjust(StitchedImage)
            if StitchedImage.shape[1]==(images[pair[0]].shape[1])or StitchedImage.shape[1]==(images[pair[1]].shape[1]):
                    StitchedImage = _stitcher(images[pair[0]], images[pair[1]])
                    StitchedImage = _adjust(StitchedImage)
            images.pop(pair[0])
            images.pop(pair[1]-1)
            index=index-1
            images.insert(0,StitchedImage)
    cv2.imwrite("Stitched.jpg", StitchedImage)
    image = StitchedImage
    original = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hsv_lower = np.array([23, 19, 141])
    hsv_upper = np.array([167, 255, 255])
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    result = cv2.bitwise_and(original, original, mask=mask)

    img = result
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15, 15))
    # plt.imshow(rgb_img)
    # plt.show()
    # grayscale..
    img = cv2.GaussianBlur(img, (7, 7), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # plt.imshow(gray)

    thresh, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    rgb_img = cv2.cvtColor(thresh_img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_img)
    plt.show()
    conts = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    print(len(conts))
    cont_img = np.zeros(img.shape)
    for c in conts:
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype='int')
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h
        if cv2.contourArea(c) < 10000 or w > 1200 or h > 430:
            continue
        print(x, y, w, h)
        cv2.drawContours(cont_img, [c], -1, (0, 255, 255, 2))
        cv2.drawContours(cont_img, [box], -1, (255, 255, 255, 1))
        approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        print(len(approx))
        if len(approx) >= 5 and aspect_ratio < 1.2:
            if w > 10 and h > 10:
                cv2.drawContours(image, [box], -1, (255, 179, 0,), 10)
                cv2.putText(image, "Sea Star", (x - 50, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)
        elif len(approx) >=5 and len(approx) < 8 and aspect_ratio > 1.2:
            if w > 10 and h > 10 and w < 700 and h < 700:
                cv2.putText(image, "Sponge", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)
                cv2.drawContours(image, [box], -1, (15, 242, 113,), 10)

        elif len(approx) == 4 and aspect_ratio < 1.5:
            cv2.putText(image, "Coral Fragment", (x - 100, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)
            cv2.drawContours(image, [box], -1, (0, 68, 255,), 10)
        elif aspect_ratio > 3:
            cv2.putText(image, "Coral Colony", (x + 100, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)
            cv2.drawContours(image, [box], -1, (12, 121, 59,), 10)
    cv2.imwrite('final.jpg', image)



