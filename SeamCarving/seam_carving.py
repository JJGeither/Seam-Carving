import cv2
import numpy as np
import sys


def compute_energy_map(img):
    r, c = img.shape
    e = np.zeros_like(img, dtype=np.uint16)
    #e(i,j) = |v(i,j)-v(i-1,j)|+ |v(i,j)-v(i+1,j)|+ |v(i,j)-v(i,j-1)|+ |v(i,j)-v(i,j+1)|

    for i in range(r):
        for j in range(c):
            up = img[i - 1, j] if i - 1 >= 0 else 0
            down = img[i + 1, j] if i + 1 < r else 0
            left = img[i, j - 1] if j - 1 >= 0 else 0
            right = img[i, j + 1] if j + 1 < c else 0

            e[i, j] = np.abs(img[i, j] - up) + np.abs(img[i, j] - down) + np.abs(img[i, j] - left) + np.abs(
                img[i, j] - right)
    return e



def calculate_cumulative(energy_map):
    r, c = energy_map.shape
    e = energy_map
    #(i = x-axis; j = y-axis):

    #M(i,j) = e(i,j) + min{M(i-1,j-1), M(i, j-1), M(i+1,j-1))


    M = energy_map.astype(np.int32).copy()
    backtrack = np.zeros_like(M, int)

    for i in range(1, r):
        for j in range(0, c):
            M[i,j] = e[i,j] + np.min(M[i - 1, j - 1], M[i, j - 1], M[i + 1, j - 1])
    return M

def find_seam(M):
    r, c = M.shape
    backtrack = np.zeros_like(M, dtype=np.int32)

    # Find the position of the smallest element in the
    # first row of M
    j = np.argmin(M[0])
    for i in range(1, r):
        for j in range(0, c):
            print(f"Pixel at ({i},{j}): {M[i, j]}")

    return backtrack


def carve_column(img):
    r, c = img.shape

    energy_map = compute_energy_map(img)
    M = calculate_cumulative(energy_map)
    backtrack = find_seam(M)

    # Create a (r, c) matrix filled with the value True
    # We'll be removing all pixels from the image which
    # have False later
    mask = np.ones((r, c), bool)

    # Find the position of the smallest element in the
    # last row of M
    j = np.argmin(M[-1])

    for i in reversed(range(r)):
        # Mark the pixels for deletion
        mask[i, j] = False
        j = backtrack[i, j]

    # Since the image has 3 channels, we convert our
    # mask to 3D
    # mask = np.stack([mask] * 3, axis=2)

    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    img = img[mask].reshape((r, c - 1))

    return img


def crop_c(img, vertSize):
    r, c = img.shape

    for i in range(vertSize):  # use range if you don't want to use tqdm
        img = carve_column(img)

    return img


if __name__ == '__main__':
    filename = sys.argv[1]
    vert_seams = int(sys.argv[2])
    horizontal_seams = int(sys.argv[3])
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = crop_c(img,vert_seams)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # flip the image 90 degrees
    img = crop_c(img, horizontal_seams)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # flip the image 90 degrees
    processed_filename = filename[:-4] + '_processed_' + str(vert_seams) + '_' + str(horizontal_seams) + '.pgm'
    cv2.imwrite(processed_filename, img)
