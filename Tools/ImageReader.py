import numpy as np
import os
import argparse
import tensorflow as tf
import re
from shutil import move
from scipy.misc import imread, imsave
from Tools.DataReader import load_labels

def image_reader(path_to_images, path_to_labels=None, save_image_as_np_path=None):
    """ """
    imglist = create_image_lists(path_to_images)
    
    # Create The Folders Corrosponding To Labels
    image_array = np.array([np.array(imread(fname)) for fname in imglist])
    if not path_to_labels is None:
        labels = load_labels(path_to_labels)
    
        if save_image_as_np_path is None:
            return image_array, labels
        else:
            path = save_image_as_np_path+"/"+"Images.npy"
            np.save(path,image_array)
            return image_array, labels
    else:
        return image_array


def create_image_lists(image_dir):
    """ Builds a list of images from the file system.
    """
    # The root directory comes first, so skip it.
    is_root_dir = True
    extensions = ['jpg', 'jpeg']
    file_list = []
    dir_name = os.path.basename(image_dir)
    tf.logging.info("Looking for images in '" + dir_name + "'")
    for extension in extensions:
        file_glob = os.path.join(image_dir, '*.' + extension)
        file_list.extend(tf.gfile.Glob(file_glob))
    if not file_list:
        tf.logging.warning('No files found')
        return

    arg_sort = np.argsort([int(re.search("(\d*\.?\d)",val).group(0)) for val in file_list])
    imglist = [file_list[idx] for idx in arg_sort]
    return imglist

if __name__ == '__main__':
    import sys
    from Tools.DataReader import load_labels

    parser = argparse.ArgumentParser(prog='Separate images into folders sorted by classes',
                                    description='''Program separates images into folders sorted by the classes of the image.
                                                The folders are created at the location specified or if unspecified at the script location''')
    parser.add_argument('path_to_image', 
                        help='Input path to images to be separated into classes')
    
    parser.add_argument('path_to_labels', 
                        help='Input path to images to be separated into classes')
    
    parser.add_argument('-path_to_save_images_npy', 
                        help='Input path to folder where the images should be saved as numpy array',
                        default=None)

    #  Parse Arguments
    args = parser.parse_args()

    # Path is a data file
    if os.path.exists(args.path_to_pictures):
        image_reader(args.path_to_pictures, args.path_to_labels, args.path_to_save_images_npy)
    else:
        print("Error: file '" + args.path_to_pictures + "' not found")
        exit(1)
    