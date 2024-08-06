
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]
SPEC_EXTENSION = '.npy'


def is_image_file(filename): # check if file is an image
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_spec_file(filename): # check if file is a spectrogram
    return filename.endswith(SPEC_EXTENSION)

def make_dataset(dir, max_dataset_size=float("inf")): # load data paths
    images = []
    spectrograms = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
            elif is_spec_file(fname):
                path = os.path.join(root, fname)
                spectrograms.append(path)

    if not images and spectrograms:
        return  spectrograms[:min(max_dataset_size, len(spectrograms))]
    elif images and not spectrograms:
        return  images[:min(max_dataset_size, len(images))]
    else:
        return images[:min(max_dataset_size, len(images))]