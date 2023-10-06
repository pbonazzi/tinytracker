import tensorflow as tf
from ITrackerDataTF import ITrackerData, dataset_preprocessing, ITrackerDataGrid, dataset_preprocessing_grid
import numpy as np
import argparse
def parse_args():
    """
    Usage: args = parse_args_train()
    """
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--data-path', type=str, help="Path to processed dataset")
    my_parser.add_argument('--batch-size', type=int, default=128)
    my_parser.add_argument('--image-size', type=int, default=112)
    my_parser.add_argument('--model-path', type=str, default=None, help="Set path of pretrained model weights")
    my_parser.add_argument('--grey', action='store_true', help="Greyscale input")
    my_parser.add_argument('--grid-embed', action='store_true', help="Grid Embedding input")
    return my_parser.parse_args()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def validate(val_loader, interpreter, criterion, epoch):
    global count_test

    losses = AverageMeter()
    lossesLin = AverageMeter()


    progbar = tf.keras.utils.Progbar(len(val_loader), stateful_metrics=['loss','lossLin'])

    oIndex = 0
    for i, data in val_loader.enumerate():
        # measure data loading time

        x_batch_val, y_batch_val = data
        x_batch_val = (x_batch_val - 128.).numpy().astype(np.int8)
        x_batch_val = np.expand_dims(x_batch_val, axis=0)
        y_batch_val = np.expand_dims(y_batch_val, axis=0)
        # compute output

        interpreter.set_tensor(input_details[0]['index'], x_batch_val)
        interpreter.invoke()
        output_details = interpreter.get_output_details()
        output = interpreter.get_tensor(output_details[0]['index'])
        output = tf.convert_to_tensor(output)*30.

        loss = criterion(output, y_batch_val)

        lossLin = output - y_batch_val
        lossLin = tf.multiply(lossLin, lossLin)
        lossLin = tf.reduce_sum(lossLin, 1)
        lossLin = tf.reduce_mean(tf.sqrt(lossLin))

        losses.update(loss.numpy(), x_batch_val.shape[0])
        lossesLin.update(lossLin.numpy(), x_batch_val.shape[0])

        # compute gradient and do SGD step
        # measure elapsed time
        progbar.update(int(i), [('loss', loss.numpy()), ('lossLin', lossLin.numpy())])


    return lossesLin.avg


args = parse_args()
shape = (args.image_size, args.image_size)
if args.grid_embed:
    dataVal = ITrackerDataGrid(args.data_path, split='val')
    dataVal =  dataset_preprocessing_grid(dataVal, 'tflite',shape, grey=args.grey)
else:
    dataVal = ITrackerData(args.data_path, split='val')
    dataVal =  dataset_preprocessing(dataVal, 'tflite',shape)
interpreter = tf.lite.Interpreter(model_path=args.model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
print(input_details)

print("Validation:",validate(dataVal, interpreter, tf.keras.losses.MeanSquaredError(),0))