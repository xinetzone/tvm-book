shape = 299, 299, 3
shape_dict = {"DecodeJpeg/contents": shape}
dtype_dict = {"DecodeJpeg/contents": "uint8"}

graph_def = create_graph(model_path)
mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)
print("TensorFlow 的 protobuf 已导入到 Relay 前端。")


import tensorflow as tf
# from tensorflow.tools.graph_transforms import TransformGraph
from graph_transforms import TransformGraph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants

def export_pb(session, model_path="myexportedmodel.pb"):
    with tf_compat_v1.gfile.GFile(model_path, "wb") as f:
        inputs = ["DecodeJpeg/contents"] # replace with your input names
        outputs = ["softmax"] # replace with your output names
        with tf_compat_v1.Session() as sess:
            graph_def = tf_testing.AddShapesToGraphDef(sess, "softmax")
        # graph_def = session.graph.as_graph_def(add_shapes=True)
        # graph_def = tf_compat_v1.graph_util.convert_variables_to_constants(session, graph_def, outputs)
        graph_def = TransformGraph(
            graph_def,
            inputs,
            outputs,
            [
                "remove_nodes(op=Identity, op=CheckNumerics, op=StopGradient)",
                "sort_by_execution_order", # sort by execution order after each transform to ensure correct node ordering
                "remove_attribute(attribute_name=_XlaSeparateCompiledGradients)",
                "remove_attribute(attribute_name=_XlaCompile)",
                "remove_attribute(attribute_name=_XlaScope)",
                "sort_by_execution_order",
                "remove_device",
                "sort_by_execution_order",
                "fold_batch_norms",
                "sort_by_execution_order",
                "fold_old_batch_norms",
                "sort_by_execution_order"
            ]
        )
        f.write(graph_def.SerializeToString())

# 从已保存的 GraphDef 创建图。
create_graph(model_path)

with tf_compat_v1.Session() as sess:
    export_pb(sess, model_path="myexportedmodel.pb")

with tf_compat_v1.gfile.GFile("myexportedmodel.pb", "rb") as f:
    graph_def = tf_compat_v1.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name="")
    # 调用实用程序将图定义导入默认图中。
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    # 在图中添加形状。
    with tf_compat_v1.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, "softmax")
shape_dict = {"DecodeJpeg/contents": x.shape}
dtype_dict = {"DecodeJpeg/contents": "uint8"}
mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)
print("TensorFlow 的 protobuf 已导入到 Relay 前端。")