import cv2
import numpy as np
# 1)
# Reads the image
img = cv2.imread('../lena-1.png', cv2.IMREAD_COLOR)

# Defining a function called padding that adds padding to the image
def padding(image, border_width):
    padded = cv2.copyMakeBorder(            # cv2.copyMakeBoarder creates a new image with a boarder
        image,
        top=border_width,
        bottom=border_width,
        left=border_width,
        right=border_width,
        borderType=cv2.BORDER_REFLECT       # Fills in the boarder as the reflection/ mirroring the image
    )
    return padded

# Calls the function adding 100 pixels as reflection border
padded_img = padding(img, 100)
# Saves the image
cv2.imwrite("Padded_lena.png", padded_img)

# 2)
def crop(image, x_0, y_0, x_1, y_1):
    h, w = image.shape[:2]
    x_start = x_0
    x_end = w - x_1
    y_start = y_0
    y_end = h - y_1
    cropped = image[y_start:y_end, x_start:x_end].copy()
    return cropped

# Cropping the image the following pixels
cropped_img = crop(img, 80, 130, 80, 130)
cv2.imwrite("Cropped_lena.png", cropped_img)



# 3)

def resize(image, width, height):
    h, w = image.shape[:2]
    resized = cv2.resize(image, (width, height))
    return resized

resized_img = resize(img, 200, 200)
cv2.imwrite("Resized_lena.png", resized_img)


import cv2
import numpy as np

img = cv2.imread('../lena-1.png', cv2.IMREAD_COLOR)


def print_image_info(image):
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    print("Height: ", height, ", Width: ", width, "Channels: ", channels)
    return height, width, channels


# 4
def copy(image, emptyPiuictureArray):
    height, width = image.shape[:2]
    for y in range(height):
        for x in range(width):
            emptyPiuictureArray[y][x] = image[y][x]     # Copy each pixel
    return emptyPiuictureArray

height, width = img.shape[:2]
emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
copied_img = copy(img, emptyPictureArray)
cv2.imwrite("copied_lena.png", copied_img)



# 5 GRAYSCALE
def grayScale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

gray_img = grayScale(img)
cv2.imwrite("Gray_lena.png", gray_img)


#6
def hsv (image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv

hsv_img = hsv(img)
cv2.imwrite("Hsv_lena.png", hsv_img)


# 7
def hue_shifted(image, emptyPictureArray, hue: int = 50):
    if emptyPictureArray is None or emptyPictureArray.size == 0 or emptyPictureArray.shape != image.shape:
        hue_s = image.copy()
    else:
        np.copyto(emptyPictureArray, image)
        hue_s = emptyPictureArray

    shifted = (hue_s.astype(np.int16) + (hue)) % 256
    shifted = shifted.astype(np.uint8)

    cv2.imwrite("Hue_shifted_lena.png", shifted)
    return shifted

height, width = image.shape[:2]
emptyHueArray = np.zeros((height, width, 3), dtype=np.uint8)
hue_img = hue_shifted(image, emptyHueArray, hue=50)



# 8
def smoothing(image):
    # Makes a 15 by 15 averging kernel
    kernel = np.ones((15, 15),dtype=np.float32) / (15*15)
    # ddepth=-1 keeps source depth
    smoothed = cv2.filter2D(image, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_DEFAULT)
    return smoothed

smoothing_img = smoothing(img)
cv2.imwrite("Smoothing_lena.png", smoothing_img)


#9
def rotation(image, rotation_angle):
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)

rot90 = rotation(img, 90)
rot180 = rotation(img, 180)

cv2.imwrite("Rotated_90.png", rot90)

cv2.imwrite("Rotated_180.png", rot180)
