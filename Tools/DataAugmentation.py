import numpy as np
import os, argparse, sys, re
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ImageReader import image_reader, create_image_lists
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf
from scipy.misc import imresize, imsave
K.set_image_dim_ordering('th')


def augment_data(path_to_images, path_to_labels, save_image_path, number_images_per_class,lbls_path_and_file_name, target_size=None, resize_only=False):
    """ """
    # Get Image List
    img_list = create_image_lists(path_to_images)
    img_name_list = [os.path.split(x)[1][:-4] for x in img_list]

    # load image data
    X_train, y_train = image_reader(path_to_images,path_to_labels)

    # resize images to target size
    if not target_size is None:
        X_train = resize_images(X_train,(target_size[0],target_size[1]))
        
        # return now if only image resizing is required
        if resize_only:
            for idx, image in enumerate(X_train):
                im_path = os.path.join(save_image_path,'Image'+str(idx+1)+'.jpg')
                imsave(im_path, image)   
            return     

    # Data Augmentation To Use
    datagen = ImageDataGenerator(
        rescale = 1./255,
        horizontal_flip = True,
        vertical_flip=True,
        fill_mode = "nearest",
        rotation_range=50,
        featurewise_center=True,
        data_format="channels_last")

    # convert from int to float
    X_train = X_train.astype('float32')

    # fit parameters from data
    datagen.fit(X_train)
    
    # Sample Count
    unique, counts = np.unique(y_train, return_counts=True)
    label_offset = int(unique[0])
    unique -= label_offset
    print("Starting Data Augmentation \n")
    for curr_class in unique:
        indices = np.where(y_train == curr_class+label_offset)[0].tolist()
        X_curr = X_train[indices,:,:,:]
        y_curr = y_train[indices]
        created_images=0
        for X_batch, y_batch in datagen.flow(X_curr, y_curr, 
                                            batch_size=10, 
                                            save_prefix="Image_C"+str(int(curr_class)),
                                            save_format="jpeg",
                                            save_to_dir=save_image_path):
            num_ = y_batch.shape
            created_images += num_[0]
            string_ = "Creating Images Of Class " + str(int(curr_class)) + ", " + str(created_images) + "/" + str(number_images_per_class) + "       "
            print(string_,end='\r',flush=True)
            if created_images >= number_images_per_class:
                break

    # Save Labels For Augmented Data
    img_list = create_image_lists(save_image_path)
    img_name_list = [os.path.split(x)[1][:-4] for x in img_list]
    new_lbls= create_labels(img_name_list)+label_offset
    np.save(lbls_path_and_file_name,new_lbls)


def augment_test_data(path_to_images, save_image_path, number_samples_per_image, target_size=None, resize_only=False):
    """ """
    # Get Image List
    img_list = create_image_lists(path_to_images)
    num_samples = len(img_list)
    img_name_list = [os.path.split(x)[1][:-4] for x in img_list]

    # load image data
    X_train = image_reader(path_to_images)

    # resize images to target size
    if not target_size is None:
        X_train = resize_images(X_train,(target_size[0],target_size[1]))
        
        # return now if only image resizing is required
        if resize_only:
            for idx, image in enumerate(X_train):
                im_path = os.path.join(save_image_path,'Image'+str(idx+1)+'.jpg')
                imsave(im_path, image)   
            return     

    # Data Augmentation To Use
    test_datagen = ImageDataGenerator(
        rescale = 1./255,
        horizontal_flip = True,
        vertical_flip=True,
        fill_mode = "nearest",
        rotation_range=50,
        featurewise_center=True,
        data_format="channels_last")

    # convert from int to float
    X_train = X_train.astype('float32')

    # fit parameters from data
    test_datagen.fit(X_train)

    print("Starting Data Augmentation \n")
    for idx, sample in enumerate(X_train):
        created_images=0
        sample = np.array([sample,sample])
        for X_batch in test_datagen.flow(sample,
                                        batch_size=1,
                                        save_prefix="Image_"+str(idx+1),
                                        save_format="jpeg",
                                        save_to_dir=save_image_path,
                                        shuffle=False):
            num, h, w, c = X_batch.shape
            created_images += num
            if created_images >= number_samples_per_image:
                break
        string_ = "Augmenting images " + str(idx+1) + "/" + str(num_samples) + "       "
        print(string_,end='\r',flush=True)


def create_labels(img_name_list):
    img_name_list = [os.path.split(x)[1][:-4] for x in img_name_list]
    labels = np.array([ int(img_lbl.split("_")[1][1:]) for img_lbl in img_name_list])
    return labels


def resize_images(image_data, target_size, interp='cubic'):
    num_samples, h, l, c = image_data.shape
    resized_data = np.zeros((num_samples, target_size[0], target_size[1], c))
    
    for i, image in enumerate(image_data):
        resized_data[i,:] = imresize(image,target_size,interp=interp)
        string_ = "Resizing images " + str(int(i+1)) + "/" + str(num_samples) + "       "
        print(string_,end='\r',flush=True)
    return resized_data

def whiten_data(path_to_images, save_image_path, epsilon):
    """ """
    # Get Image List
    img_list = create_image_lists(path_to_images)
    img_name_list = [os.path.split(x)[1][:-4] for x in img_list]

    # load image data
    X_train = image_reader(path_to_images) 

    X_train_zca = zca_whitening(X_train, epsilon=epsilon, zero_centered=False)
    for idx, image in enumerate(X_train_zca):
        im_path = os.path.join(save_image_path,'Image'+str(idx+1)+'.jpg')
        imsave(im_path, image)


# Attempt at homebrewing a ZCA whitening transform
# https://stackoverflow.com/questions/41635737/is-this-the-correct-way-of-whitening-an-image-in-python
def zca_whitening(X, epsilon=1e-6, zero_centered=True):
    num_samples, h, w, c = X.shape
    
    # zero center data if not already zero-centered
    if not zero_centered:
        X = X - X.mean(axis=0)

    # reshape data into single feature vectors (N,h*w*c)
    X = X.reshape(num_samples, h*w*c)
    
    # compute the covariance of the image data
    cov = np.cov(X, rowvar=True)   # cov is (N, N)

    # apply singular value decomposition
    U,S,V = np.linalg.svd(cov)     # U is (N, N), S is (N,)

    # build the ZCA matrix
    zca_matrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))      # zca_matrix is (N,N)
    
    # apply zca transform
    zca = np.dot(zca_matrix, X)     # zca is (N, h*w*c)
    
    # reshape zca transform back into image shape (N,h,w,c)
    zca = zca.reshape(num_samples, h, w, c)
    
    return zca
    

if __name__ == '__main__':
    import sys
    from Tools.DataReader import load_labels

    parser = argparse.ArgumentParser(prog='Augments Images and saves them on the local drive',
                                    description='''Program augments images and saves them on the local drive, in a specified folder.
                                                The specified number of images per class will approximatly be created
                                                Images are saved in the specified folder''')
    # train data augmentation
    parser.add_argument('-input_data', 
                        help='Input path to images to be augmented',
                        default='Data/Train/TrainImages')
    
    parser.add_argument('-input_label', 
                        help='Input path to images to be separated into classes',
                        default='Data/Train/trainLbls.txt')
    
    parser.add_argument('-output_dir', 
                        help='Output path to folder where the images should be saved',
                        required=True)
    
    parser.add_argument('-images_per_class',
                        help='Total number of images that will be within each class (approximately)',
                        type=int,
                        default=2000)

    parser.add_argument('-output_lbls',
                        help='File name of augmented data labels numpy file',
                        default="Lbls")

    parser.add_argument('-target_size',
                        help='Output image dimensions (h, w)',
                        nargs=2,
                        type=int,
                        required=False)

    parser.add_argument('-resize_only', 
                        help='Only perform image resizing - no other augmentation',
                        action="store_true")

    parser.add_argument('-whiten', 
                        help='Apply ZCA whitening to input images and store in output folder',
                        action="store_true")

    parser.add_argument('-epsilon',
                        help='Epsilon parameter for ZCA whitening',
                        type=float,
                        default=1e-6)

    # test data augmentation
    parser.add_argument('-test_data', 
                        help='Data to be augmented is test data',
                        action="store_true")

    parser.add_argument('-images_per_sample',
                    help='Number of augmented samples to be created for each test image',
                    type=int,
                    default=4)

    #  Parse Arguments
    args = parser.parse_args()

    if os.path.exists(args.input_data):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir )
        
        if args.whiten:
            whiten_data(args.input_data, args.output_dir, args.epsilon)
        else:
            if args.test_data:
                augment_test_data(args.input_data, args.output_dir, args.images_per_sample, target_size=args.target_size, resize_only=args.resize_only)
            else:
                augment_data(args.input_data, args.input_label, args.output_dir, args.images_per_class,args.output_lbls, target_size=args.target_size, resize_only=args.resize_only)
    else:
        print("Error: file '" + args.input_data + "' not found")
        exit(1)