import cv2

# load a color image in grayscale
img = cv2.imread('lena-1.png', cv2.IMREAD_COLOR)

cv2.waitKey(0)
cv2.destroyAllWindows()

def print_image_information(image):

    # Dimensions
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]

    print("Image dimensions: ")
    print("Height: ", height)
    print("Width: ", width)
    print("Size: ", image.size) # heigth * width * channels
    print("Channels: ", channels)
    print("Data type: ", image.dtype)

# Calling the function
print_image_information(img)

