import tensorflow as tf
import tf2onnx

model = tf.keras.models.load_model('digit_recognition_model.h5')
spec = (tf.TensorSpec((None, 10, 15, 1), tf.float32, name="input"),)
output_path = "digit_recognition_model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
