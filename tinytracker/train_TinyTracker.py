from src.TinyTrackerTF import TinyTracker
import tensorflow as tf
from tensorflow import keras
from src.ITrackerDataTF import ITrackerData, dataset_preprocessing, dataset_preprocessing_grid, ITrackerDataGrid

import argparse
import src.json_utils as json_utils
def parse_args():
    """
    Usage: args = parse_args_train()
    """
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--data-path', type=str, help="Path to processed dataset")
    my_parser.add_argument('--num-epochs', type=int, default=15)
    my_parser.add_argument('--batch-size', type=int, default=128)
    my_parser.add_argument('--image-size', type=int, default=112)
    my_parser.add_argument('--learning-rate', type=float, default=0.001)
    my_parser.add_argument('--pretrained', type=str, default=None, help="Set path of pretrained model weights")
    my_parser.add_argument('--backbone', type=str, default="mobilenetv3", help="Set path of pretrained model weights")
    my_parser.add_argument('--grid-emb', action='store_true', help="Enable grid embedding input")
    my_parser.add_argument('--grey', action='store_true', help="use greyscale images as input")
    return my_parser.parse_args()


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
@tf.function
def train_step(x_batch_train, y_batch_train):
    with tf.GradientTape() as tape:

        logits = model(x_batch_train, training=True) * 30.

        loss_value = loss_fn(y_batch_train, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value

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
def validate(val_loader, model, criterion, epoch):

    losses = AverageMeter()
    lossesLin = AverageMeter()

    # switch to evaluate mode
    model.trainable = False

    progbar = tf.keras.utils.Progbar(len(val_loader), stateful_metrics=['loss'])

    oIndex = 0
    for i, data in val_loader.enumerate():
        x_batch_val, y_batch_val = data
        output = model(x_batch_val, training=False) * 30.
        loss = criterion(output, y_batch_val)

        lossLin = output - y_batch_val
        lossLin = tf.multiply(lossLin, lossLin)
        lossLin = tf.reduce_sum(lossLin, 1)
        lossLin = tf.reduce_mean(tf.sqrt(lossLin))

        losses.update(loss.numpy(), x_batch_val.shape[0])
        lossesLin.update(lossLin.numpy(), x_batch_val.shape[0])

        # compute gradient and do SGD step
        # measure elapsed time
        progbar.update(int(i), [('loss', loss.numpy())])
    final_loss = lossesLin.avg
    losses.reset()
    lossesLin.reset()

    return final_loss


info = {}
args = parse_args()
print(args)
info['args'] = args

results = {}
if args.grey:
    in_channels = 1
else:
    in_channels = 3

if args.grid_emb:
    in_channels += 2

imSize = (args.image_size,args.image_size)
batch_size = args.batch_size
epochs = args.num_epochs


#MODEL
model = TinyTracker(in_channels=in_channels, backbone=args.backbone)
print("TinyTracker number of input channels:", in_channels)
print("TinyTracker backbone:", args.backbone)
model.trainable = True
model((tf.random.uniform((1, args.image_size, args.image_size, in_channels))))
stringlist = []
model.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)
info['summary'] = short_model_summary
print(short_model_summary)
if args.pretrained:
    try:
        model.trainable = True
        model.load_weights(args.pretrained)
    except:
        print("Loading model with trainable=True failed, loading with trainable=False instead")
        model.trainable = False
        model.load_weights(args.pretrained)

#DATACONFIGURATION

name = "TinyTracker_"+str(imSize[0])+"_"+ str(in_channels)

#OPTIMIZER
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=args.learning_rate,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
loss_fn = keras.losses.MeanSquaredError()
best_loss = 10000
#DATASETS
data_path = args.data_path

if args.grid_emb:
    train_dataset = ITrackerDataGrid(data_path, split='train')
    train_dataset = train_dataset.shuffle(100*batch_size, reshuffle_each_iteration=True)
    train_dataset = dataset_preprocessing_grid(train_dataset, 'train', imSize, grey=args.grey).batch(batch_size)
    dataVal = ITrackerDataGrid(data_path, split='val')
    dataVal =  dataset_preprocessing_grid(dataVal, 'val', imSize, grey=args.grey).batch(batch_size)
else:
    train_dataset = ITrackerData(data_path, split='train')
    train_dataset = train_dataset.shuffle(100*batch_size, reshuffle_each_iteration=True)
    train_dataset = dataset_preprocessing(train_dataset, 'train', imSize).batch(batch_size)
    dataVal = ITrackerData(data_path, split='val')
    dataVal =  dataset_preprocessing(dataVal, 'val', imSize).batch(batch_size)

if args.pretrained:
    p_val_loss =validate(dataVal, model, loss_fn, 0)
    print("\n Pretrained Validation:", p_val_loss)
    info['pretrained_val_loss'] = p_val_loss

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    progbar = tf.keras.utils.Progbar(len(train_dataset), stateful_metrics=['loss'])
    model.trainable = True

    for step, data in enumerate(train_dataset):
        x_batch_train, y_batch_train = data
        loss_value = train_step(x_batch_train, y_batch_train)
        progbar.update(step, [('loss', loss_value)])


    val_loss = validate(dataVal, model, loss_fn, epoch)
    results[epoch] = val_loss
    model.save_weights(name + "_check.hdf5")
    info['results'] = results
    if val_loss < best_loss:
        model.save_weights(name+".hdf5")
        best_loss = val_loss
        json_utils.save_json(name + ".json", results)

    print("\nValidation:", val_loss)
json_utils.save_json(name+".json", results)



