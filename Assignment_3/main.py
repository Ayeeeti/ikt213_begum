import cv2
import numpy as np

img_lambo = cv2.imread('Outputs/lambo.png', cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img_lambo, cv2.COLOR_BGR2GRAY)

# 1
def sobel_edge_detection(image):
    # Convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Bluring the image
    blurred = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)
    # Sobel with float depth, dx=1, dy=1, ksize=1
    sobel = cv2.Sobel(blurred, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)
    return sobel

sobel_edge_img = sobel_edge_detection(img_lambo)
cv2.imwrite("sobel_edges.png", cv2.convertScaleAbs(sobel_edge_img))


# 2
def canny_edge_detection(image, threshold_1, threshold_2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)
    edges = cv2.Canny(blurred, threshold_1, threshold_2)
    return edges

canny_edges_img = canny_edge_detection(img_lambo, 50, 50)
cv2.imwrite("Outputs/canny_edges.png", canny_edges_img)


# 3
img_shapes = cv2.imread('shapes-1.png', cv2.IMREAD_COLOR)
img_shapes_template = cv2.imread('shapes_template.jpg', cv2.IMREAD_COLOR)

def template_match(image, template):
    threshold = 0.9

    # gray scale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # normalized correlation
    res = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)                     # Get location above threshold

    # Draw rectangles on outout
    out = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)     # Convertes back to 3-channel image
    h, w = tpl_gray.shape[:2]                            # Gives the height and width of the template
    for pt in zip(*loc[::-1]):                           # pt is the top-left corner of each detection. zip(*location[::-1]) flips them into (x, y) pairs
        cv2.rectangle(out, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)      # Draws green rectangles the same size as the template

    cv2.imwrite("shapes_template.png", out)
    return out

template_match_img = template_match(img_shapes, img_shapes_template)
cv2.imwrite("Outputs/template_match.jpg", template_match_img)



# 4
def resize(image_lambo, scale_factor: int, up_or_down: str):
    out = image_lambo.copy()

    if up_or_down == 'up':
        for _ in range(scale_factor // 2):
                out = cv2.pyrUp(out)
    elif up_or_down == 'down':
        for _ in range(scale_factor // 2):
            out = cv2.pyrDown(out)

    image_resize = f"resize_{up_or_down}_{scale_factor}.png"
    cv2.imwrite(image_resize, out)
    return out

resized_lambo = resize(img_lambo, scale_factor=2, up_or_down='down')
resize_lambo = resize(img_lambo, scale_factor=2, up_or_down='up')




