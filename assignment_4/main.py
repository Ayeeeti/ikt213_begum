import cv2
import numpy as np

img_r = cv2.imread('reference_img.png')

# 1 Harris Corner Detection:
def harris_corner_detection(reference_img):
    #ref_img = cv2.imread(reference_img)
    gray = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    kernel = np.ones((2, 2), np.uint8)
    dst = cv2.dilate(dst, kernel)

    out = reference_img.copy()
    out[dst > 0.01 * dst.max()] = [0, 0, 255]

    return out
harris = harris_corner_detection(img_r)
cv2.imwrite('harris.png',harris)

def align_image(align_this, reference_image, max_features=10, good_match_precent=0.7):

    img_a = cv2.imread(align_this, cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(reference_image, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()
    kp_a, des_a = sift.detectAndCompute(img_a, None)
    kp_r, des_r = sift.detectAndCompute(img_r, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_a, des_r, k=2)


    good_match = []
    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good_match.append(m)

    if len(good_match) > max_features:
        src_pts = np.float32([ kp_a[m.queryIdx].pt for m in good_match]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_r[m.trainIdx].pt for m in good_match]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img_a.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)


        aligned = cv2.warpPerspective(img_a, M, (img_r.shape[1], img_r.shape[0]))
        cv2.imwrite('aligned.png', aligned)

    draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = matchesMask, flags = 2)
    img_c = cv2.drawMatches(img_a, kp_a, img_r, kp_r, good_match, None, **draw_params)

    cv2.imwrite('matches.png', img_c)
align_image('align_this.jpg', 'reference_img.png')