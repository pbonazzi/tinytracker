import tensorflow as tf
import scipy.io as sio
import os
import os.path
import numpy as np
import random
MEAN_PATH = './'

def loadMetadata(filename, silent=False):
    try:
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata


def SubtractMean(image, meanImg):
    meanImg = tf.convert_to_tensor(meanImg / 255, dtype=tf.float32)
    image = tf.cast(image, tf.float32)
    return tf.math.subtract(image, meanImg)


def loadImage(path, channels=3):
    try:
        im = tf.io.read_file(path)
        im = tf.image.decode_jpeg(im, channels=channels)
    except tf.errors.NotFoundError:
        raise RuntimeError('Could not read image: ' + path)

    return im


def makeGrid(params, gridSize):
    gridLen = gridSize[0] * gridSize[1]
    grid = np.zeros([gridLen, ], np.float32)

    indsY = np.array([i // gridSize[0] for i in range(gridLen)])
    indsX = np.array([i % gridSize[0] for i in range(gridLen)])
    condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2])
    condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3])
    cond = np.logical_and(condX, condY)

    grid[cond] = 1
    return tf.convert_to_tensor(grid, dtype=tf.float32)



# def load_and_preprocess_dataset(index, metadata, dataPath, imSize):
#     index = metadata['labelRecNum'][index]
#
#     imFacePath = os.path.join(dataPath, '%05d/appleFace/%05d.jpg' % (
#     metadata['labelRecNum'][index], metadata['frameIndex'][index]))
#
#     imFace = preprocess_image(imFacePath, imSize)
#
#     gaze = np.array([metadata['labelDotXCam'][index], metadata['labelDotYCam'][index]], np.float32)
#
#
#     return tf.cast(index, tf.int64), imFace, tf.convert_to_tensor(gaze, dtype=tf.float32)
#

def ITrackerData(dataPath, split='train'):
    print('Loading iTracker dataset...')
    metaFile = os.path.join(dataPath, 'metadata.mat')






    if metaFile is None or not os.path.isfile(metaFile):
        raise RuntimeError('There is no such file %s! Provide a valid dataset path.' % metaFile)

    metadata = loadMetadata(metaFile)
    if metadata is None:
        raise RuntimeError('Could not read metadata file %s! Provide a valid dataset path.' % metaFile)

    if split == 'test':
        mask = metadata['labelTest']
    elif split == 'val':
        mask = metadata['labelVal']
    else:
        mask = metadata['labelTrain']

    indices = np.argwhere(mask)[:, 0]
    paths = [os.path.join(dataPath,'%05d/appleFace/%05d.jpg' % (metadata['labelRecNum'][i], metadata['frameIndex'][i])) for i in indices]
    gaze = [[metadata['labelDotXCam'][i], metadata['labelDotYCam'][i]] for i in indices]



    print('Loaded iTracker dataset split "%s" with %d records...' % (split, len(indices)))



    dataset = tf.data.Dataset.from_tensor_slices((paths, gaze))

    #dataset = dataset_preprocessing(dataset,split)

    return dataset

def ITrackerDataGrid(dataPath, split='train'):
    print('Loading iTracker dataset...')
    metaFile = os.path.join(dataPath, 'metadata.mat')

    if metaFile is None or not os.path.isfile(metaFile):
        raise RuntimeError('There is no such file %s! Provide a valid dataset path.' % metaFile)

    metadata = loadMetadata(metaFile)
    if metadata is None:
        raise RuntimeError('Could not read metadata file %s! Provide a valid dataset path.' % metaFile)

    if split == 'test':
        mask = metadata['labelTest']
    elif split == 'val':
        mask = metadata['labelVal']
    else:
        mask = metadata['labelTrain']

    indices = np.argwhere(mask)[:, 0]
    paths = [os.path.join(dataPath,'%05d/appleFace/%05d.jpg' % (metadata['labelRecNum'][i], metadata['frameIndex'][i])) for i in indices]
    gaze = [[metadata['labelDotXCam'][i], metadata['labelDotYCam'][i]] for i in indices]
    grid = [metadata['labelFaceGrid'][i,:] for i in indices]


    print('Loaded iTracker dataset split "%s" with %d records...' % (split, len(indices)))



    dataset = tf.data.Dataset.from_tensor_slices((paths, gaze, grid))

    #dataset = dataset_preprocessing(dataset,split)

    return dataset
def create_grid(g, sx, sy):
    x_start = g[0]*2/25 - 1
    x_end = (g[0] + g[2])*2/25-1
    y_start = g[1]*2/25 - 1
    y_end = (g[1] + g[3])*2/25-1
    linx = tf.linspace(x_start,x_end,sx)
    liny = tf.linspace(y_start,y_end,sy)
    return tf.meshgrid(linx, liny)
def dataset_preprocessing(dataset, split, shape):
    @tf.function
    def random_augmentations(image):
        image = image + tf.random.normal(image.shape, 0, 3) * tf.random.normal([1], 0, 1)
        image = tf.image.random_brightness(image, 0.5)
        image = tf.image.random_contrast(image, 0.5, 1.5)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.clip_by_value(image, 0, 255)
        image = tf.scalar_mul(1 / 127.5, image)
        image = tf.add(image, -1.)
        return image
    @tf.function
    def preprocess_image_tflite(image_path, gaze):
        image = loadImage(image_path)
        image = tf.image.resize(image, shape)
        return image, tf.cast(gaze, tf.float32)

    @tf.function
    def preprocess_image_train(image_path, gaze):
        image = loadImage(image_path)
        image = tf.image.resize(image, shape)
        image = random_augmentations(image)
        return image, tf.cast(gaze, tf.float32)

    @tf.function
    def preprocess_image(image_path, gaze):
        image = loadImage(image_path)
        image = tf.image.resize(image, shape)
        return image / 127.5 - 1, tf.cast(gaze, tf.float32)

    if split == 'train':
        dataset = dataset.map(
            lambda path, gaze: preprocess_image_train(path, gaze))
    elif split == 'tflite':
        dataset = dataset.map(
            lambda path, gaze: preprocess_image_tflite(path, gaze))
    else:
        dataset = dataset.map(
            lambda path, gaze: preprocess_image(path, gaze))

    return dataset.prefetch(tf.data.AUTOTUNE)

def dataset_preprocessing_grid(dataset, split, shape, grey=False):
    if grey:
        channels = 1

        @tf.function
        def random_augmentations(image):
            image = image + tf.random.normal(image.shape, 0, 3) * tf.random.normal([1], 0, 1)
            image = tf.image.random_brightness(image, 0.5)
            image = tf.image.random_contrast(image, 0.5, 1.5)
            image = tf.clip_by_value(image, 0, 255)
            image = tf.scalar_mul(1 / 127.5, image)
            image = tf.add(image, -1.)
            return image

    else:
        channels = 3

        @tf.function
        def random_augmentations(image):
            image = image + tf.random.normal(image.shape, 0, 3) * tf.random.normal([1], 0, 1)
            image = tf.image.random_brightness(image, 0.5)
            image = tf.image.random_contrast(image, 0.5, 1.5)
            image = tf.image.random_saturation(image, 0.8, 1.2)
            image = tf.image.random_hue(image, max_delta=0.05)
            image = tf.clip_by_value(image, 0, 255)
            image = tf.scalar_mul(1 / 127.5, image)
            image = tf.add(image, -1.)
            return image



    @tf.function
    def preprocess_image_grid(image_path, gaze, grid):
        image = loadImage(image_path, channels=channels)
        image = tf.image.resize(image, shape)
        image = image / 127.5 - 1
        gridX, gridY = create_grid(grid,shape[0],shape[1])
        gridX = tf.expand_dims(gridX, axis=-1)
        gridY = tf.expand_dims(gridY, axis=-1)
        image = tf.concat([image, gridX, gridY], axis=-1)
        return image, tf.cast(gaze, tf.float32)

    @tf.function
    def preprocess_image_grid_tflite(image_path, gaze, grid):
        image = loadImage(image_path, channels=channels)
        image = tf.image.resize(image, shape)
        gridX, gridY = create_grid(grid,shape[0],shape[1])
        gridX = tf.expand_dims(gridX, axis=-1) * 127.5 + 128
        gridY = tf.expand_dims(gridY, axis=-1) * 127.5 + 128
        image = tf.concat([image, gridX, gridY], axis=-1)
        return image, tf.cast(gaze, tf.float32)

    @tf.function
    def preprocess_image_grid_train(image_path, gaze, grid):
        image = loadImage(image_path, channels=channels)
        image = tf.image.resize(image, shape)
        image = random_augmentations(image)
        gridX, gridY = create_grid(grid,shape[0],shape[1])
        gridX = tf.expand_dims(gridX, axis=-1)
        gridY = tf.expand_dims(gridY, axis=-1)
        image = tf.concat([image, gridX, gridY], axis=-1)
        return image, tf.cast(gaze, tf.float32)


    if split == 'train':
        dataset = dataset.map(
            lambda path, gaze, grid: preprocess_image_grid_train(path, gaze,grid))
    elif split == 'tflite':
        dataset = dataset.map(
            lambda path, gaze,grid: preprocess_image_grid_tflite(path, gaze,grid))
    else:
        dataset = dataset.map(
            lambda path, gaze,grid: preprocess_image_grid(path, gaze,grid))

    return dataset.prefetch(tf.data.AUTOTUNE)
