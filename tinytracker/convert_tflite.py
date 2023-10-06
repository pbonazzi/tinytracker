
from src.TinyTrackerTF import TinyTracker, bakedModel
from src.TinyTrackerS import lightweight_mobilenetv3
import tensorflow as tf
import argparse

def parse_args():
    """
    Usage: args = parse_args_train()
    """
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--model-path', type=str, default="TinyTracker.hdf5")
    my_parser.add_argument('--name', type=str, default="TinyTracker")
    my_parser.add_argument('--image-size', type=int, default=112)
    my_parser.add_argument('--learning-rate', type=float, default=0.001)
    my_parser.add_argument('--grid-emb', action='store_true', help="Enable grid embedding input")
    my_parser.add_argument('--grey', action='store_true', help="use greyscale images as input")
    my_parser.add_argument('--bake', action='store_true', help="Bake grid embedding into tflite model")
    return my_parser.parse_args()

args = parse_args()

grid_embed = args.grid_emb
grey = args.grey
shape = (args.image_size,args.image_size)
# selects amount of input channels (default should be 3)
# Only if grid embedding and RGB is enabled 5 will be required
if args.grey:
    in_channels = 1
else:
    in_channels = 3
if args.grid_emb:
    in_channels += 2


if args.bake:
    k_model = bakedModel(lightweight_mobilenetv3((shape[0], shape[1], in_channels), 2), shape=shape)
else:
    k_model = TinyTracker(grid_embed=grid_embed and not grey)

k_model((tf.random.uniform((1, shape[0], shape[1], in_channels))))
k_model.trainable = False
k_model.model.load_weights(args.model_path)
def convert(model, output_type = tf.float32, n_samples = 500):

    if grid_embed:
        def representative_data_gen():
            from src.ITrackerDataTF import  ITrackerDataGrid, dataset_preprocessing_grid
            train_dataset =  ITrackerDataGrid(r"F:\GazeCapture\Processed", split="val")
            train_dataset = train_dataset.shuffle(n_samples)
            train_dataset = dataset_preprocessing_grid(train_dataset, 'val', shape, grey=grey)
            for image,_ in train_dataset.take(n_samples):
                if args.bake and grey:
                    yield [tf.expand_dims(image[:,:,:1], axis=0)]
                else:
                    yield [tf.expand_dims(image,axis=0)]
    else:
        def representative_data_gen():
            from src.ITrackerDataTF import  ITrackerData, dataset_preprocessing
            train_dataset =  ITrackerData(r"F:\GazeCapture\Processed", split="val")
            train_dataset = train_dataset.shuffle(n_samples)
            train_dataset = dataset_preprocessing(train_dataset, 'val', shape)
            for image,_ in train_dataset.take(n_samples):
                yield [tf.expand_dims(image,axis=0)]


    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    converter.representative_dataset = representative_data_gen
    converter.inference_type = tf.int8
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = output_type  # or tf.uint8

    return converter.convert()


tflite = convert(k_model)
# converter = tf.lite.TFLiteConverter.from_keras_model(k_model)
# tflite = converter.convert()


with open(args.model_+".tflite", 'wb') as f:
    f.write(tflite)
    print("Model written to:" + args.name+".tflite")

print("done")
#with torch.no_grad(): torch.onnx.export(model, torch.rand((1,3,128,128)), "test.onnx", opset_version=12,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)