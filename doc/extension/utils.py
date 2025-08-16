import onnx
import onnxruntime

def run_onnx_model(model_path, inputs):
    """获取 ONNX 推理结果"""
    onnx_model = onnx.load(model_path)
    # 通过 ONNX 运行模型以获取预期结果
    ort_session = onnxruntime.InferenceSession(
        onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return ort_session.run([], inputs)