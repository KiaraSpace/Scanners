import numpy as np
import cv2 as cv


def unsharp_mask(im, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    bl = cv.GaussianBlur(im, kernel_size, sigma)
    sharpened = float(amount + 1) * im - float(amount) * bl
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(im - bl) < threshold
        np.copyto(sharpened, im, where=low_contrast_mask)
    return sharpened
