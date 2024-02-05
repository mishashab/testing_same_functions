import tensorflow
from tf2onnx
import os
import onnx

IN_MODEL_PATH = '..\\models\\learned_on_linux'
OUT_MODEL_PATH = '..\\models\\onnx_learned_on_linux'


def load_model(model_path):
    return tensorflow.keras.models.load_model(model_path)


def translate_model_from_tf_to_onnx(in_model_path=IN_MODEL_PATH,
                                    out_model_path=OUT_MODEL_PATH):

    model = load_model(in_model_path)
    input_signature = [tensorflow.TensorSpec([3, 3], tensorflow.float32, name='x')]
    onnx_model, _ = tf2onnx.convert(model, opset=13)
    onnx.save(onnx_model, os.path.join(out_model_path,'model_onnx'))


if __name__ == '__main__':
    translate_model_from_tf_to_onnx()
