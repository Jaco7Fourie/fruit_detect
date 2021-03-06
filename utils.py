import skimage
import skimage.io
import skimage.transform
import numpy as np
from PIL import ImageFont, ImageDraw, Image


# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of desired shape
# [height, width, depth]



def load_image(path, new_size):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to new_size
    resized_img = skimage.transform.resize(crop_img, new_size)
    return resized_img

def write_string_to_image(string, image, font_size=25):
    font = ImageFont.truetype("DroidSans.ttf", font_size)
    pil_im = Image.fromarray(np.uint8(image*255))
    draw = ImageDraw.Draw(pil_im)
    draw.text((0, 0), string, (255, 255, 255), font=font)
    return np.array(pil_im)

def draw_box_to_image(image, rgb_col, coordinates):
    pil_im = Image.fromarray(np.uint8(image*255))
    draw = ImageDraw.Draw(pil_im)
    draw.rectangle(coordinates, outline=rgb_col)
    return np.array(pil_im)

def crop_to_coords(im, xStart, xEnd, yStart, yEnd, new_size):
    """
        crop image to the coordinates given and resize to new_size for training
    :return: the cropped image of shape new_size
    """
    cropped = im[yStart:yEnd+1, xStart:xEnd+1]
    # we crop image from center
    #short_edge = min(cropped.shape[:2])
    #yy = int((cropped.shape[0] - short_edge) / 2)
    #xx = int((cropped.shape[1] - short_edge) / 2)
    #crop_img = cropped[yy: yy + short_edge, xx: xx + short_edge]
    # resize to new_size
    resized_img = skimage.transform.resize(cropped, new_size, order=3, mode='reflect')    # use bi-cubic since bi-qudratic seems to be broken
    return resized_img

# [height, width, depth]
def load_image_without_resize(path):
    # load image
    img = skimage.io.imread(path, plugin='matplotlib')
    if ((img > 1.0).any()):
        img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    return img


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)


if __name__ == "__main__":
    test()
