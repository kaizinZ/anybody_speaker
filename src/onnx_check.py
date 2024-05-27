import onnx
import onnxruntime as rt
session = rt.InferenceSession("./onnx_models/model.onnx", providers=["CPUExecutionProvider"])