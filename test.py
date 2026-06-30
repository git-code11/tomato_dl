import tensorflow as tf
from tomato_dl.utils.tflite import TfliteInference
from pprint import pprint


CLASS_NAMES = ["Vegetative", "Flowering", "Fruiting"]
IMAGE_PATH = "./flowering_data_0046.jpg"
IMAGE_SIZE = (256, 256)


def load_image(img_path: str):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img


hybrid_effnet = TfliteInference(
    model_path=r"./hybrid_xception.tflite", labels=CLASS_NAMES)
hybrid_effnet.load_model()

# load image
img = load_image(IMAGE_PATH)
# normalize image
img = img / 255.
# expand image dims
img = tf.expand_dims(img, axis=0)
# perform inference
result = hybrid_effnet.inference([img])
pprint(result['labelled'])
