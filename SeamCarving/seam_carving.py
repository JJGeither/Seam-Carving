import cv2
import numpy as np
import sys


def compute_energy_map(img):
    r, c = img.shape
    e = np.zeros_like(img, dtype=np.uint32)

    for j in range(0, c - 1):
        for i in range(0, r - 1):

           # print(f"Pixel value at ({i}, {j}): {img[i, j]}; Image value: {img[i, j]}")
            left = img[i - 1, j] if i > 0 else 0
            right = img[i + 1, j] if i < r - 1 else 0
            up = img[i, j - 1] if j > 0 else 0
            down = img[i, j + 1] if j < c - 1 else 0
            e[i, j] = np.abs(img[i, j] - up) + np.abs(img[i, j] - down) + np.abs(img[i, j] - left) + np.abs(img[i, j] - right)
            #print(f"Pixel value at ({i}, {j}): {img[i, j]}; Energy value: {e[i, j]}")

    return e


def calculate_cumulative(energy_map):
    r, c = energy_map.shape
    e = energy_map
    #(i = x-axis; j = y-axis):

    #M(i,j) = e(i,j) + min{M(i-1,j-1), M(i, j-1), M(i+1,j-1))
    M = energy_map.astype(np.int32).copy()
    backtrack = np.zeros_like(M, int)

    for j in range(c - 1):
        for i in range(r - 1):
            M[i,j] = e[i,j] + np.min([M[i - 1, j - 1], M[i, j - 1], M[i + 1, j - 1]])

    # Find the minimum value in the first row of M
    min_val = np.min(M[:, c - 1])
    min_index = np.argmin(M[:, c - 1])
    min_coord = (min_index, c - 1)
    backtrack[min_coord] = 100
    print(f"Minimum pixel value at ({min_coord}): Energy value: {min_val}")
    print(f"{r} x {c}")

    #Now continue creating the seam

    for j in range(r ):
        print(f"{j} : THIS")
        below = (j, min_coord[1])
        belowLeft = (j, min_coord[1] - 1) if min_coord[1] - 1 > 0 else below
        belowRight = (j, min_coord[1] + 1) if min_coord[1] + 1 < 0 else below
        min_coord = belowLeft
        if (M[below] < M[min_coord]) :
            min_coord = below
        if (M[belowRight] < M[min_coord]):
            min_coord = belowRight

        backtrack[min_coord] = 100;

    processed_filename = filename[:-4] + '_seam_' + str(vert_seams) + '_' + str(horizontal_seams) + '.pgm'
    cv2.imwrite(processed_filename, backtrack)

    return M


def find_seam(M):
    r, c = M.shape
    backtrack = np.zeros_like(M, dtype=np.int32)

    # Find the position of the smallest element in the
    # first row of M


    return backtrack

def carve_column(img):
    r, c = img.shape

    print("Energymap")
    energy_map = compute_energy_map(img)
    print("Energymap Completed")
    print("Cumulative")
    M = calculate_cumulative(energy_map)
    print("Cumulative Completed")
    print("Backtrack")
    backtrack = find_seam(M)
    print("Backtrack Complete")

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
