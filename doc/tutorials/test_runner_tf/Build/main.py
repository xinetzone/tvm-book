import os
import numpy as np
from typing import List, Dict, Any
import tvm
from tvm.contrib.msc.core import utils as msc_utils
from tensorflow.python import ops
from tvm.contrib.msc.framework.tensorflow import tf_v1

# Define the helpers
def load_data(name: str, shape: List[int], dtype: str) -> np.ndarray:
  path = os.path.join("baseline", name + ".bin")
  if os.path.isfile(path):
    data = np.fromfile(path, dtype=dtype).reshape(shape)
  else:
    data = np.ones((shape)).astype(dtype)
  return data

def get_variable(name: str, shape: List[int], dtype: str, weights: Dict[str, tvm.nd.array]) -> tf_v1.Tensor:
  if name in weights:
    var = tf_v1.get_variable(name, initializer=weights[name].asnumpy())
  else:
    var = tf_v1.get_variable(name, shape, dtype)
  return var


# Define the graph
def main(res_0: tf_v1.Tensor, weights: Dict[str, tvm.nd.array]) -> List[tf_v1.Tensor]:
  # Define the weights
  weight_1 = get_variable("const_70", [3, 3, 3, 48], "float32", weights)
  bias_1 = get_variable("const", [48], "float32", weights)
  weight_3 = get_variable("const_72", [3, 3, 48, 1], "float32", weights)
  weight_7 = get_variable("const_68", [1, 1, 48, 24], "float32", weights)
  bias_7 = get_variable("const_1", [24], "float32", weights)
  weight_8 = get_variable("const_66", [1, 1, 24, 144], "float32", weights)
  bias_8 = get_variable("const_2", [144], "float32", weights)
  weight_10 = get_variable("const_77", [3, 3, 144, 1], "float32", weights)
  weight_14 = get_variable("const_64", [1, 1, 144, 32], "float32", weights)
  bias_14 = get_variable("const_3", [32], "float32", weights)
  weight_15 = get_variable("const_62", [1, 1, 32, 192], "float32", weights)
  bias_15 = get_variable("const_4", [192], "float32", weights)
  weight_17 = get_variable("const_82", [3, 3, 192, 1], "float32", weights)
  weight_21 = get_variable("const_60", [1, 1, 192, 32], "float32", weights)
  bias_21 = get_variable("const_5", [32], "float32", weights)
  weight_23 = get_variable("const_58", [1, 1, 32, 192], "float32", weights)
  bias_23 = get_variable("const_6", [192], "float32", weights)
  weight_25 = get_variable("const_87", [3, 3, 192, 1], "float32", weights)
  weight_29 = get_variable("const_56", [1, 1, 192, 48], "float32", weights)
  bias_29 = get_variable("const_7", [48], "float32", weights)
  weight_30 = get_variable("const_54", [1, 1, 48, 288], "float32", weights)
  bias_30 = get_variable("const_8", [288], "float32", weights)
  weight_32 = get_variable("const_92", [3, 3, 288, 1], "float32", weights)
  weight_36 = get_variable("const_52", [1, 1, 288, 48], "float32", weights)
  bias_36 = get_variable("const_9", [48], "float32", weights)
  weight_38 = get_variable("const_50", [1, 1, 48, 288], "float32", weights)
  bias_38 = get_variable("const_10", [288], "float32", weights)
  weight_40 = get_variable("const_97", [3, 3, 288, 1], "float32", weights)
  weight_44 = get_variable("const_48", [1, 1, 288, 48], "float32", weights)
  bias_44 = get_variable("const_11", [48], "float32", weights)
  weight_46 = get_variable("const_46", [1, 1, 48, 288], "float32", weights)
  bias_46 = get_variable("const_12", [288], "float32", weights)
  weight_48 = get_variable("const_102", [3, 3, 288, 1], "float32", weights)
  weight_52 = get_variable("const_44", [1, 1, 288, 88], "float32", weights)
  bias_52 = get_variable("const_13", [88], "float32", weights)
  weight_53 = get_variable("const_42", [1, 1, 88, 528], "float32", weights)
  bias_53 = get_variable("const_14", [528], "float32", weights)
  weight_55 = get_variable("const_107", [3, 3, 528, 1], "float32", weights)
  weight_59 = get_variable("const_40", [1, 1, 528, 88], "float32", weights)
  bias_59 = get_variable("const_15", [88], "float32", weights)
  weight_61 = get_variable("const_38", [1, 1, 88, 528], "float32", weights)
  bias_61 = get_variable("const_16", [528], "float32", weights)
  weight_63 = get_variable("const_112", [3, 3, 528, 1], "float32", weights)
  weight_67 = get_variable("const_36", [1, 1, 528, 88], "float32", weights)
  bias_67 = get_variable("const_17", [88], "float32", weights)
  weight_69 = get_variable("const_34", [1, 1, 88, 528], "float32", weights)
  bias_69 = get_variable("const_18", [528], "float32", weights)
  weight_71 = get_variable("const_117", [3, 3, 528, 1], "float32", weights)
  weight_75 = get_variable("const_32", [1, 1, 528, 88], "float32", weights)
  bias_75 = get_variable("const_19", [88], "float32", weights)
  weight_77 = get_variable("const_30", [1, 1, 88, 528], "float32", weights)
  bias_77 = get_variable("const_20", [528], "float32", weights)
  weight_79 = get_variable("const_122", [3, 3, 528, 1], "float32", weights)
  weight_83 = get_variable("const_28", [1, 1, 528, 136], "float32", weights)
  bias_83 = get_variable("const_21", [136], "float32", weights)
  weight_84 = get_variable("const_26", [1, 1, 136, 816], "float32", weights)
  bias_84 = get_variable("const_22", [816], "float32", weights)
  weight_86 = get_variable("const_127", [3, 3, 816, 1], "float32", weights)
  weight_90 = get_variable("const_24", [1, 1, 816, 136], "float32", weights)
  bias_90 = get_variable("const_23", [136], "float32", weights)
  weight_92 = get_variable("const_22_1", [1, 1, 136, 816], "float32", weights)
  bias_92 = get_variable("const_25", [816], "float32", weights)
  weight_94 = get_variable("const_132", [3, 3, 816, 1], "float32", weights)
  weight_98 = get_variable("const_20_1", [1, 1, 816, 136], "float32", weights)
  bias_98 = get_variable("const_27", [136], "float32", weights)
  weight_100 = get_variable("const_18_1", [1, 1, 136, 816], "float32", weights)
  bias_100 = get_variable("const_29", [816], "float32", weights)
  weight_102 = get_variable("const_137", [3, 3, 816, 1], "float32", weights)
  weight_106 = get_variable("const_16_1", [1, 1, 816, 224], "float32", weights)
  bias_106 = get_variable("const_31", [224], "float32", weights)
  weight_107 = get_variable("const_14_1", [1, 1, 224, 1344], "float32", weights)
  bias_107 = get_variable("const_33", [1344], "float32", weights)
  weight_109 = get_variable("const_142", [3, 3, 1344, 1], "float32", weights)
  weight_113 = get_variable("const_12_1", [1, 1, 1344, 224], "float32", weights)
  bias_113 = get_variable("const_35", [224], "float32", weights)
  weight_115 = get_variable("const_10_1", [1, 1, 224, 1344], "float32", weights)
  bias_115 = get_variable("const_37", [1344], "float32", weights)
  weight_117 = get_variable("const_147", [3, 3, 1344, 1], "float32", weights)
  weight_121 = get_variable("const_8_1", [1, 1, 1344, 224], "float32", weights)
  bias_121 = get_variable("const_39", [224], "float32", weights)
  weight_123 = get_variable("const_6_1", [1, 1, 224, 1344], "float32", weights)
  bias_123 = get_variable("const_41", [1344], "float32", weights)
  weight_125 = get_variable("const_152", [3, 3, 1344, 1], "float32", weights)
  weight_129 = get_variable("const_4_1", [1, 1, 1344, 448], "float32", weights)
  bias_129 = get_variable("const_43", [448], "float32", weights)
  weight_130 = get_variable("const_2_1", [1, 1, 448, 1792], "float32", weights)
  bias_130 = get_variable("const_45", [1792], "float32", weights)
  weight_133 = get_variable("const_47", [1, 1, 1792, 1001], "float32", weights)
  bias_133 = get_variable("const_49", [1001], "float32", weights)
  # Define the ops
  # gv(msc.conv2d_bias): <res_0> -> <res_1>
  res_1 = ops.nn_ops.conv2d(res_0, weight_1, strides=[1, 2, 2, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv")
  res_1 = ops.nn_ops.bias_add(res_1, bias_1, name="gv_bias")
  # MobilenetV2_Conv_Relu6(clip): <res_1> -> <res_2>
  res_2 = tf_v1.clip_by_value(res_1, clip_value_min=0, clip_value_max=6, name="MobilenetV2_Conv_Relu6")
  # MobilenetV2_expanded_conv_depthwise_depthwise(nn.conv2d): <res_2> -> <res_3>
  res_3 = ops.nn_ops.depthwise_conv2d_native(res_2, weight_3, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="MobilenetV2_expanded_conv_depthwise_depthwise")
  # MobilenetV2_expanded_conv_depthwise_BatchNorm_FusedBatchNorm(nn.batch_norm): <res_3> -> <res_4>
  res_4 = tf_v1.layers.batch_normalization(res_3, scale=True, center=True, momentum=0.1, epsilon=0.001, gamma_initializer=tf_v1.constant_initializer(weights["const_73"].asnumpy()), beta_initializer=tf_v1.constant_initializer(weights["const_74"].asnumpy()), moving_mean_initializer=tf_v1.constant_initializer(weights["const_75"].asnumpy()), moving_variance_initializer=tf_v1.constant_initializer(weights["const_76"].asnumpy()), name="MobilenetV2_expanded_conv_depthwise_BatchNorm_FusedBatchNorm")
  # MobilenetV2_expanded_conv_depthwise_BatchNorm_FusedBatchNorm.0(get_item): <res_4> -> <res_5>
  res_5 = res_4
  # MobilenetV2_expanded_conv_depthwise_Relu6(clip): <res_5> -> <res_6>
  res_6 = tf_v1.clip_by_value(res_5, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_depthwise_Relu6")
  # gv2(msc.conv2d_bias): <res_6> -> <res_7>
  res_7 = ops.nn_ops.conv2d(res_6, weight_7, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv2")
  res_7 = ops.nn_ops.bias_add(res_7, bias_7, name="gv2_bias")
  # gv4(msc.conv2d_bias): <res_7> -> <res_8>
  res_8 = ops.nn_ops.conv2d(res_7, weight_8, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv4")
  res_8 = ops.nn_ops.bias_add(res_8, bias_8, name="gv4_bias")
  # MobilenetV2_expanded_conv_1_expand_Relu6(clip): <res_8> -> <res_9>
  res_9 = tf_v1.clip_by_value(res_8, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_1_expand_Relu6")
  # MobilenetV2_expanded_conv_1_depthwise_depthwise(nn.conv2d): <res_9> -> <res_10>
  res_10 = ops.nn_ops.depthwise_conv2d_native(res_9, weight_10, strides=[1, 2, 2, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="MobilenetV2_expanded_conv_1_depthwise_depthwise")
  # MobilenetV2_expanded_conv_1_depthwise_BatchNorm_FusedBatchNorm(nn.batch_norm): <res_10> -> <res_11>
  res_11 = tf_v1.layers.batch_normalization(res_10, scale=True, center=True, momentum=0.1, epsilon=0.001, gamma_initializer=tf_v1.constant_initializer(weights["const_78"].asnumpy()), beta_initializer=tf_v1.constant_initializer(weights["const_79"].asnumpy()), moving_mean_initializer=tf_v1.constant_initializer(weights["const_80"].asnumpy()), moving_variance_initializer=tf_v1.constant_initializer(weights["const_81"].asnumpy()), name="MobilenetV2_expanded_conv_1_depthwise_BatchNorm_FusedBatchNorm")
  # MobilenetV2_expanded_conv_1_depthwise_BatchNorm_FusedBatchNorm.0(get_item): <res_11> -> <res_12>
  res_12 = res_11
  # MobilenetV2_expanded_conv_1_depthwise_Relu6(clip): <res_12> -> <res_13>
  res_13 = tf_v1.clip_by_value(res_12, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_1_depthwise_Relu6")
  # gv6(msc.conv2d_bias): <res_13> -> <res_14>
  res_14 = ops.nn_ops.conv2d(res_13, weight_14, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv6")
  res_14 = ops.nn_ops.bias_add(res_14, bias_14, name="gv6_bias")
  # gv8(msc.conv2d_bias): <res_14> -> <res_15>
  res_15 = ops.nn_ops.conv2d(res_14, weight_15, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv8")
  res_15 = ops.nn_ops.bias_add(res_15, bias_15, name="gv8_bias")
  # MobilenetV2_expanded_conv_2_expand_Relu6(clip): <res_15> -> <res_16>
  res_16 = tf_v1.clip_by_value(res_15, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_2_expand_Relu6")
  # MobilenetV2_expanded_conv_2_depthwise_depthwise(nn.conv2d): <res_16> -> <res_17>
  res_17 = ops.nn_ops.depthwise_conv2d_native(res_16, weight_17, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="MobilenetV2_expanded_conv_2_depthwise_depthwise")
  # MobilenetV2_expanded_conv_2_depthwise_BatchNorm_FusedBatchNorm(nn.batch_norm): <res_17> -> <res_18>
  res_18 = tf_v1.layers.batch_normalization(res_17, scale=True, center=True, momentum=0.1, epsilon=0.001, gamma_initializer=tf_v1.constant_initializer(weights["const_83"].asnumpy()), beta_initializer=tf_v1.constant_initializer(weights["const_84"].asnumpy()), moving_mean_initializer=tf_v1.constant_initializer(weights["const_85"].asnumpy()), moving_variance_initializer=tf_v1.constant_initializer(weights["const_86"].asnumpy()), name="MobilenetV2_expanded_conv_2_depthwise_BatchNorm_FusedBatchNorm")
  # MobilenetV2_expanded_conv_2_depthwise_BatchNorm_FusedBatchNorm.0(get_item): <res_18> -> <res_19>
  res_19 = res_18
  # MobilenetV2_expanded_conv_2_depthwise_Relu6(clip): <res_19> -> <res_20>
  res_20 = tf_v1.clip_by_value(res_19, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_2_depthwise_Relu6")
  # gv10(msc.conv2d_bias): <res_20> -> <res_21>
  res_21 = ops.nn_ops.conv2d(res_20, weight_21, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv10")
  res_21 = ops.nn_ops.bias_add(res_21, bias_21, name="gv10_bias")
  # MobilenetV2_expanded_conv_2_add(add): <res_21,res_14> -> <res_22>
  res_22 = tf_v1.add(res_21, res_14, name="MobilenetV2_expanded_conv_2_add")
  # gv12(msc.conv2d_bias): <res_22> -> <res_23>
  res_23 = ops.nn_ops.conv2d(res_22, weight_23, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv12")
  res_23 = ops.nn_ops.bias_add(res_23, bias_23, name="gv12_bias")
  # MobilenetV2_expanded_conv_3_expand_Relu6(clip): <res_23> -> <res_24>
  res_24 = tf_v1.clip_by_value(res_23, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_3_expand_Relu6")
  # MobilenetV2_expanded_conv_3_depthwise_depthwise(nn.conv2d): <res_24> -> <res_25>
  res_25 = ops.nn_ops.depthwise_conv2d_native(res_24, weight_25, strides=[1, 2, 2, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="MobilenetV2_expanded_conv_3_depthwise_depthwise")
  # MobilenetV2_expanded_conv_3_depthwise_BatchNorm_FusedBatchNorm(nn.batch_norm): <res_25> -> <res_26>
  res_26 = tf_v1.layers.batch_normalization(res_25, scale=True, center=True, momentum=0.1, epsilon=0.001, gamma_initializer=tf_v1.constant_initializer(weights["const_88"].asnumpy()), beta_initializer=tf_v1.constant_initializer(weights["const_89"].asnumpy()), moving_mean_initializer=tf_v1.constant_initializer(weights["const_90"].asnumpy()), moving_variance_initializer=tf_v1.constant_initializer(weights["const_91"].asnumpy()), name="MobilenetV2_expanded_conv_3_depthwise_BatchNorm_FusedBatchNorm")
  # MobilenetV2_expanded_conv_3_depthwise_BatchNorm_FusedBatchNorm.0(get_item): <res_26> -> <res_27>
  res_27 = res_26
  # MobilenetV2_expanded_conv_3_depthwise_Relu6(clip): <res_27> -> <res_28>
  res_28 = tf_v1.clip_by_value(res_27, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_3_depthwise_Relu6")
  # gv14(msc.conv2d_bias): <res_28> -> <res_29>
  res_29 = ops.nn_ops.conv2d(res_28, weight_29, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv14")
  res_29 = ops.nn_ops.bias_add(res_29, bias_29, name="gv14_bias")
  # gv16(msc.conv2d_bias): <res_29> -> <res_30>
  res_30 = ops.nn_ops.conv2d(res_29, weight_30, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv16")
  res_30 = ops.nn_ops.bias_add(res_30, bias_30, name="gv16_bias")
  # MobilenetV2_expanded_conv_4_expand_Relu6(clip): <res_30> -> <res_31>
  res_31 = tf_v1.clip_by_value(res_30, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_4_expand_Relu6")
  # MobilenetV2_expanded_conv_4_depthwise_depthwise(nn.conv2d): <res_31> -> <res_32>
  res_32 = ops.nn_ops.depthwise_conv2d_native(res_31, weight_32, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="MobilenetV2_expanded_conv_4_depthwise_depthwise")
  # MobilenetV2_expanded_conv_4_depthwise_BatchNorm_FusedBatchNorm(nn.batch_norm): <res_32> -> <res_33>
  res_33 = tf_v1.layers.batch_normalization(res_32, scale=True, center=True, momentum=0.1, epsilon=0.001, gamma_initializer=tf_v1.constant_initializer(weights["const_93"].asnumpy()), beta_initializer=tf_v1.constant_initializer(weights["const_94"].asnumpy()), moving_mean_initializer=tf_v1.constant_initializer(weights["const_95"].asnumpy()), moving_variance_initializer=tf_v1.constant_initializer(weights["const_96"].asnumpy()), name="MobilenetV2_expanded_conv_4_depthwise_BatchNorm_FusedBatchNorm")
  # MobilenetV2_expanded_conv_4_depthwise_BatchNorm_FusedBatchNorm.0(get_item): <res_33> -> <res_34>
  res_34 = res_33
  # MobilenetV2_expanded_conv_4_depthwise_Relu6(clip): <res_34> -> <res_35>
  res_35 = tf_v1.clip_by_value(res_34, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_4_depthwise_Relu6")
  # gv18(msc.conv2d_bias): <res_35> -> <res_36>
  res_36 = ops.nn_ops.conv2d(res_35, weight_36, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv18")
  res_36 = ops.nn_ops.bias_add(res_36, bias_36, name="gv18_bias")
  # MobilenetV2_expanded_conv_4_add(add): <res_36,res_29> -> <res_37>
  res_37 = tf_v1.add(res_36, res_29, name="MobilenetV2_expanded_conv_4_add")
  # gv20(msc.conv2d_bias): <res_37> -> <res_38>
  res_38 = ops.nn_ops.conv2d(res_37, weight_38, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv20")
  res_38 = ops.nn_ops.bias_add(res_38, bias_38, name="gv20_bias")
  # MobilenetV2_expanded_conv_5_expand_Relu6(clip): <res_38> -> <res_39>
  res_39 = tf_v1.clip_by_value(res_38, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_5_expand_Relu6")
  # MobilenetV2_expanded_conv_5_depthwise_depthwise(nn.conv2d): <res_39> -> <res_40>
  res_40 = ops.nn_ops.depthwise_conv2d_native(res_39, weight_40, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="MobilenetV2_expanded_conv_5_depthwise_depthwise")
  # MobilenetV2_expanded_conv_5_depthwise_BatchNorm_FusedBatchNorm(nn.batch_norm): <res_40> -> <res_41>
  res_41 = tf_v1.layers.batch_normalization(res_40, scale=True, center=True, momentum=0.1, epsilon=0.001, gamma_initializer=tf_v1.constant_initializer(weights["const_98"].asnumpy()), beta_initializer=tf_v1.constant_initializer(weights["const_99"].asnumpy()), moving_mean_initializer=tf_v1.constant_initializer(weights["const_100"].asnumpy()), moving_variance_initializer=tf_v1.constant_initializer(weights["const_101"].asnumpy()), name="MobilenetV2_expanded_conv_5_depthwise_BatchNorm_FusedBatchNorm")
  # MobilenetV2_expanded_conv_5_depthwise_BatchNorm_FusedBatchNorm.0(get_item): <res_41> -> <res_42>
  res_42 = res_41
  # MobilenetV2_expanded_conv_5_depthwise_Relu6(clip): <res_42> -> <res_43>
  res_43 = tf_v1.clip_by_value(res_42, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_5_depthwise_Relu6")
  # gv22(msc.conv2d_bias): <res_43> -> <res_44>
  res_44 = ops.nn_ops.conv2d(res_43, weight_44, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv22")
  res_44 = ops.nn_ops.bias_add(res_44, bias_44, name="gv22_bias")
  # MobilenetV2_expanded_conv_5_add(add): <res_44,res_37> -> <res_45>
  res_45 = tf_v1.add(res_44, res_37, name="MobilenetV2_expanded_conv_5_add")
  # gv24(msc.conv2d_bias): <res_45> -> <res_46>
  res_46 = ops.nn_ops.conv2d(res_45, weight_46, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv24")
  res_46 = ops.nn_ops.bias_add(res_46, bias_46, name="gv24_bias")
  # MobilenetV2_expanded_conv_6_expand_Relu6(clip): <res_46> -> <res_47>
  res_47 = tf_v1.clip_by_value(res_46, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_6_expand_Relu6")
  # MobilenetV2_expanded_conv_6_depthwise_depthwise(nn.conv2d): <res_47> -> <res_48>
  res_48 = ops.nn_ops.depthwise_conv2d_native(res_47, weight_48, strides=[1, 2, 2, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="MobilenetV2_expanded_conv_6_depthwise_depthwise")
  # MobilenetV2_expanded_conv_6_depthwise_BatchNorm_FusedBatchNorm(nn.batch_norm): <res_48> -> <res_49>
  res_49 = tf_v1.layers.batch_normalization(res_48, scale=True, center=True, momentum=0.1, epsilon=0.001, gamma_initializer=tf_v1.constant_initializer(weights["const_103"].asnumpy()), beta_initializer=tf_v1.constant_initializer(weights["const_104"].asnumpy()), moving_mean_initializer=tf_v1.constant_initializer(weights["const_105"].asnumpy()), moving_variance_initializer=tf_v1.constant_initializer(weights["const_106"].asnumpy()), name="MobilenetV2_expanded_conv_6_depthwise_BatchNorm_FusedBatchNorm")
  # MobilenetV2_expanded_conv_6_depthwise_BatchNorm_FusedBatchNorm.0(get_item): <res_49> -> <res_50>
  res_50 = res_49
  # MobilenetV2_expanded_conv_6_depthwise_Relu6(clip): <res_50> -> <res_51>
  res_51 = tf_v1.clip_by_value(res_50, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_6_depthwise_Relu6")
  # gv26(msc.conv2d_bias): <res_51> -> <res_52>
  res_52 = ops.nn_ops.conv2d(res_51, weight_52, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv26")
  res_52 = ops.nn_ops.bias_add(res_52, bias_52, name="gv26_bias")
  # gv28(msc.conv2d_bias): <res_52> -> <res_53>
  res_53 = ops.nn_ops.conv2d(res_52, weight_53, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv28")
  res_53 = ops.nn_ops.bias_add(res_53, bias_53, name="gv28_bias")
  # MobilenetV2_expanded_conv_7_expand_Relu6(clip): <res_53> -> <res_54>
  res_54 = tf_v1.clip_by_value(res_53, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_7_expand_Relu6")
  # MobilenetV2_expanded_conv_7_depthwise_depthwise(nn.conv2d): <res_54> -> <res_55>
  res_55 = ops.nn_ops.depthwise_conv2d_native(res_54, weight_55, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="MobilenetV2_expanded_conv_7_depthwise_depthwise")
  # MobilenetV2_expanded_conv_7_depthwise_BatchNorm_FusedBatchNorm(nn.batch_norm): <res_55> -> <res_56>
  res_56 = tf_v1.layers.batch_normalization(res_55, scale=True, center=True, momentum=0.1, epsilon=0.001, gamma_initializer=tf_v1.constant_initializer(weights["const_108"].asnumpy()), beta_initializer=tf_v1.constant_initializer(weights["const_109"].asnumpy()), moving_mean_initializer=tf_v1.constant_initializer(weights["const_110"].asnumpy()), moving_variance_initializer=tf_v1.constant_initializer(weights["const_111"].asnumpy()), name="MobilenetV2_expanded_conv_7_depthwise_BatchNorm_FusedBatchNorm")
  # MobilenetV2_expanded_conv_7_depthwise_BatchNorm_FusedBatchNorm.0(get_item): <res_56> -> <res_57>
  res_57 = res_56
  # MobilenetV2_expanded_conv_7_depthwise_Relu6(clip): <res_57> -> <res_58>
  res_58 = tf_v1.clip_by_value(res_57, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_7_depthwise_Relu6")
  # gv30(msc.conv2d_bias): <res_58> -> <res_59>
  res_59 = ops.nn_ops.conv2d(res_58, weight_59, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv30")
  res_59 = ops.nn_ops.bias_add(res_59, bias_59, name="gv30_bias")
  # MobilenetV2_expanded_conv_7_add(add): <res_59,res_52> -> <res_60>
  res_60 = tf_v1.add(res_59, res_52, name="MobilenetV2_expanded_conv_7_add")
  # gv32(msc.conv2d_bias): <res_60> -> <res_61>
  res_61 = ops.nn_ops.conv2d(res_60, weight_61, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv32")
  res_61 = ops.nn_ops.bias_add(res_61, bias_61, name="gv32_bias")
  # MobilenetV2_expanded_conv_8_expand_Relu6(clip): <res_61> -> <res_62>
  res_62 = tf_v1.clip_by_value(res_61, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_8_expand_Relu6")
  # MobilenetV2_expanded_conv_8_depthwise_depthwise(nn.conv2d): <res_62> -> <res_63>
  res_63 = ops.nn_ops.depthwise_conv2d_native(res_62, weight_63, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="MobilenetV2_expanded_conv_8_depthwise_depthwise")
  # MobilenetV2_expanded_conv_8_depthwise_BatchNorm_FusedBatchNorm(nn.batch_norm): <res_63> -> <res_64>
  res_64 = tf_v1.layers.batch_normalization(res_63, scale=True, center=True, momentum=0.1, epsilon=0.001, gamma_initializer=tf_v1.constant_initializer(weights["const_113"].asnumpy()), beta_initializer=tf_v1.constant_initializer(weights["const_114"].asnumpy()), moving_mean_initializer=tf_v1.constant_initializer(weights["const_115"].asnumpy()), moving_variance_initializer=tf_v1.constant_initializer(weights["const_116"].asnumpy()), name="MobilenetV2_expanded_conv_8_depthwise_BatchNorm_FusedBatchNorm")
  # MobilenetV2_expanded_conv_8_depthwise_BatchNorm_FusedBatchNorm.0(get_item): <res_64> -> <res_65>
  res_65 = res_64
  # MobilenetV2_expanded_conv_8_depthwise_Relu6(clip): <res_65> -> <res_66>
  res_66 = tf_v1.clip_by_value(res_65, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_8_depthwise_Relu6")
  # gv34(msc.conv2d_bias): <res_66> -> <res_67>
  res_67 = ops.nn_ops.conv2d(res_66, weight_67, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv34")
  res_67 = ops.nn_ops.bias_add(res_67, bias_67, name="gv34_bias")
  # MobilenetV2_expanded_conv_8_add(add): <res_67,res_60> -> <res_68>
  res_68 = tf_v1.add(res_67, res_60, name="MobilenetV2_expanded_conv_8_add")
  # gv36(msc.conv2d_bias): <res_68> -> <res_69>
  res_69 = ops.nn_ops.conv2d(res_68, weight_69, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv36")
  res_69 = ops.nn_ops.bias_add(res_69, bias_69, name="gv36_bias")
  # MobilenetV2_expanded_conv_9_expand_Relu6(clip): <res_69> -> <res_70>
  res_70 = tf_v1.clip_by_value(res_69, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_9_expand_Relu6")
  # MobilenetV2_expanded_conv_9_depthwise_depthwise(nn.conv2d): <res_70> -> <res_71>
  res_71 = ops.nn_ops.depthwise_conv2d_native(res_70, weight_71, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="MobilenetV2_expanded_conv_9_depthwise_depthwise")
  # MobilenetV2_expanded_conv_9_depthwise_BatchNorm_FusedBatchNorm(nn.batch_norm): <res_71> -> <res_72>
  res_72 = tf_v1.layers.batch_normalization(res_71, scale=True, center=True, momentum=0.1, epsilon=0.001, gamma_initializer=tf_v1.constant_initializer(weights["const_118"].asnumpy()), beta_initializer=tf_v1.constant_initializer(weights["const_119"].asnumpy()), moving_mean_initializer=tf_v1.constant_initializer(weights["const_120"].asnumpy()), moving_variance_initializer=tf_v1.constant_initializer(weights["const_121"].asnumpy()), name="MobilenetV2_expanded_conv_9_depthwise_BatchNorm_FusedBatchNorm")
  # MobilenetV2_expanded_conv_9_depthwise_BatchNorm_FusedBatchNorm.0(get_item): <res_72> -> <res_73>
  res_73 = res_72
  # MobilenetV2_expanded_conv_9_depthwise_Relu6(clip): <res_73> -> <res_74>
  res_74 = tf_v1.clip_by_value(res_73, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_9_depthwise_Relu6")
  # gv38(msc.conv2d_bias): <res_74> -> <res_75>
  res_75 = ops.nn_ops.conv2d(res_74, weight_75, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv38")
  res_75 = ops.nn_ops.bias_add(res_75, bias_75, name="gv38_bias")
  # MobilenetV2_expanded_conv_9_add(add): <res_75,res_68> -> <res_76>
  res_76 = tf_v1.add(res_75, res_68, name="MobilenetV2_expanded_conv_9_add")
  # gv40(msc.conv2d_bias): <res_76> -> <res_77>
  res_77 = ops.nn_ops.conv2d(res_76, weight_77, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv40")
  res_77 = ops.nn_ops.bias_add(res_77, bias_77, name="gv40_bias")
  # MobilenetV2_expanded_conv_10_expand_Relu6(clip): <res_77> -> <res_78>
  res_78 = tf_v1.clip_by_value(res_77, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_10_expand_Relu6")
  # MobilenetV2_expanded_conv_10_depthwise_depthwise(nn.conv2d): <res_78> -> <res_79>
  res_79 = ops.nn_ops.depthwise_conv2d_native(res_78, weight_79, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="MobilenetV2_expanded_conv_10_depthwise_depthwise")
  # MobilenetV2_expanded_conv_10_depthwise_BatchNorm_FusedBatchNorm(nn.batch_norm): <res_79> -> <res_80>
  res_80 = tf_v1.layers.batch_normalization(res_79, scale=True, center=True, momentum=0.1, epsilon=0.001, gamma_initializer=tf_v1.constant_initializer(weights["const_123"].asnumpy()), beta_initializer=tf_v1.constant_initializer(weights["const_124"].asnumpy()), moving_mean_initializer=tf_v1.constant_initializer(weights["const_125"].asnumpy()), moving_variance_initializer=tf_v1.constant_initializer(weights["const_126"].asnumpy()), name="MobilenetV2_expanded_conv_10_depthwise_BatchNorm_FusedBatchNorm")
  # MobilenetV2_expanded_conv_10_depthwise_BatchNorm_FusedBatchNorm.0(get_item): <res_80> -> <res_81>
  res_81 = res_80
  # MobilenetV2_expanded_conv_10_depthwise_Relu6(clip): <res_81> -> <res_82>
  res_82 = tf_v1.clip_by_value(res_81, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_10_depthwise_Relu6")
  # gv42(msc.conv2d_bias): <res_82> -> <res_83>
  res_83 = ops.nn_ops.conv2d(res_82, weight_83, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv42")
  res_83 = ops.nn_ops.bias_add(res_83, bias_83, name="gv42_bias")
  # gv44(msc.conv2d_bias): <res_83> -> <res_84>
  res_84 = ops.nn_ops.conv2d(res_83, weight_84, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv44")
  res_84 = ops.nn_ops.bias_add(res_84, bias_84, name="gv44_bias")
  # MobilenetV2_expanded_conv_11_expand_Relu6(clip): <res_84> -> <res_85>
  res_85 = tf_v1.clip_by_value(res_84, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_11_expand_Relu6")
  # MobilenetV2_expanded_conv_11_depthwise_depthwise(nn.conv2d): <res_85> -> <res_86>
  res_86 = ops.nn_ops.depthwise_conv2d_native(res_85, weight_86, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="MobilenetV2_expanded_conv_11_depthwise_depthwise")
  # MobilenetV2_expanded_conv_11_depthwise_BatchNorm_FusedBatchNorm(nn.batch_norm): <res_86> -> <res_87>
  res_87 = tf_v1.layers.batch_normalization(res_86, scale=True, center=True, momentum=0.1, epsilon=0.001, gamma_initializer=tf_v1.constant_initializer(weights["const_128"].asnumpy()), beta_initializer=tf_v1.constant_initializer(weights["const_129"].asnumpy()), moving_mean_initializer=tf_v1.constant_initializer(weights["const_130"].asnumpy()), moving_variance_initializer=tf_v1.constant_initializer(weights["const_131"].asnumpy()), name="MobilenetV2_expanded_conv_11_depthwise_BatchNorm_FusedBatchNorm")
  # MobilenetV2_expanded_conv_11_depthwise_BatchNorm_FusedBatchNorm.0(get_item): <res_87> -> <res_88>
  res_88 = res_87
  # MobilenetV2_expanded_conv_11_depthwise_Relu6(clip): <res_88> -> <res_89>
  res_89 = tf_v1.clip_by_value(res_88, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_11_depthwise_Relu6")
  # gv46(msc.conv2d_bias): <res_89> -> <res_90>
  res_90 = ops.nn_ops.conv2d(res_89, weight_90, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv46")
  res_90 = ops.nn_ops.bias_add(res_90, bias_90, name="gv46_bias")
  # MobilenetV2_expanded_conv_11_add(add): <res_90,res_83> -> <res_91>
  res_91 = tf_v1.add(res_90, res_83, name="MobilenetV2_expanded_conv_11_add")
  # gv48(msc.conv2d_bias): <res_91> -> <res_92>
  res_92 = ops.nn_ops.conv2d(res_91, weight_92, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv48")
  res_92 = ops.nn_ops.bias_add(res_92, bias_92, name="gv48_bias")
  # MobilenetV2_expanded_conv_12_expand_Relu6(clip): <res_92> -> <res_93>
  res_93 = tf_v1.clip_by_value(res_92, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_12_expand_Relu6")
  # MobilenetV2_expanded_conv_12_depthwise_depthwise(nn.conv2d): <res_93> -> <res_94>
  res_94 = ops.nn_ops.depthwise_conv2d_native(res_93, weight_94, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="MobilenetV2_expanded_conv_12_depthwise_depthwise")
  # MobilenetV2_expanded_conv_12_depthwise_BatchNorm_FusedBatchNorm(nn.batch_norm): <res_94> -> <res_95>
  res_95 = tf_v1.layers.batch_normalization(res_94, scale=True, center=True, momentum=0.1, epsilon=0.001, gamma_initializer=tf_v1.constant_initializer(weights["const_133"].asnumpy()), beta_initializer=tf_v1.constant_initializer(weights["const_134"].asnumpy()), moving_mean_initializer=tf_v1.constant_initializer(weights["const_135"].asnumpy()), moving_variance_initializer=tf_v1.constant_initializer(weights["const_136"].asnumpy()), name="MobilenetV2_expanded_conv_12_depthwise_BatchNorm_FusedBatchNorm")
  # MobilenetV2_expanded_conv_12_depthwise_BatchNorm_FusedBatchNorm.0(get_item): <res_95> -> <res_96>
  res_96 = res_95
  # MobilenetV2_expanded_conv_12_depthwise_Relu6(clip): <res_96> -> <res_97>
  res_97 = tf_v1.clip_by_value(res_96, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_12_depthwise_Relu6")
  # gv50(msc.conv2d_bias): <res_97> -> <res_98>
  res_98 = ops.nn_ops.conv2d(res_97, weight_98, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv50")
  res_98 = ops.nn_ops.bias_add(res_98, bias_98, name="gv50_bias")
  # MobilenetV2_expanded_conv_12_add(add): <res_98,res_91> -> <res_99>
  res_99 = tf_v1.add(res_98, res_91, name="MobilenetV2_expanded_conv_12_add")
  # gv52(msc.conv2d_bias): <res_99> -> <res_100>
  res_100 = ops.nn_ops.conv2d(res_99, weight_100, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv52")
  res_100 = ops.nn_ops.bias_add(res_100, bias_100, name="gv52_bias")
  # MobilenetV2_expanded_conv_13_expand_Relu6(clip): <res_100> -> <res_101>
  res_101 = tf_v1.clip_by_value(res_100, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_13_expand_Relu6")
  # MobilenetV2_expanded_conv_13_depthwise_depthwise(nn.conv2d): <res_101> -> <res_102>
  res_102 = ops.nn_ops.depthwise_conv2d_native(res_101, weight_102, strides=[1, 2, 2, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="MobilenetV2_expanded_conv_13_depthwise_depthwise")
  # MobilenetV2_expanded_conv_13_depthwise_BatchNorm_FusedBatchNorm(nn.batch_norm): <res_102> -> <res_103>
  res_103 = tf_v1.layers.batch_normalization(res_102, scale=True, center=True, momentum=0.1, epsilon=0.001, gamma_initializer=tf_v1.constant_initializer(weights["const_138"].asnumpy()), beta_initializer=tf_v1.constant_initializer(weights["const_139"].asnumpy()), moving_mean_initializer=tf_v1.constant_initializer(weights["const_140"].asnumpy()), moving_variance_initializer=tf_v1.constant_initializer(weights["const_141"].asnumpy()), name="MobilenetV2_expanded_conv_13_depthwise_BatchNorm_FusedBatchNorm")
  # MobilenetV2_expanded_conv_13_depthwise_BatchNorm_FusedBatchNorm.0(get_item): <res_103> -> <res_104>
  res_104 = res_103
  # MobilenetV2_expanded_conv_13_depthwise_Relu6(clip): <res_104> -> <res_105>
  res_105 = tf_v1.clip_by_value(res_104, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_13_depthwise_Relu6")
  # gv54(msc.conv2d_bias): <res_105> -> <res_106>
  res_106 = ops.nn_ops.conv2d(res_105, weight_106, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv54")
  res_106 = ops.nn_ops.bias_add(res_106, bias_106, name="gv54_bias")
  # gv56(msc.conv2d_bias): <res_106> -> <res_107>
  res_107 = ops.nn_ops.conv2d(res_106, weight_107, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv56")
  res_107 = ops.nn_ops.bias_add(res_107, bias_107, name="gv56_bias")
  # MobilenetV2_expanded_conv_14_expand_Relu6(clip): <res_107> -> <res_108>
  res_108 = tf_v1.clip_by_value(res_107, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_14_expand_Relu6")
  # MobilenetV2_expanded_conv_14_depthwise_depthwise(nn.conv2d): <res_108> -> <res_109>
  res_109 = ops.nn_ops.depthwise_conv2d_native(res_108, weight_109, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="MobilenetV2_expanded_conv_14_depthwise_depthwise")
  # MobilenetV2_expanded_conv_14_depthwise_BatchNorm_FusedBatchNorm(nn.batch_norm): <res_109> -> <res_110>
  res_110 = tf_v1.layers.batch_normalization(res_109, scale=True, center=True, momentum=0.1, epsilon=0.001, gamma_initializer=tf_v1.constant_initializer(weights["const_143"].asnumpy()), beta_initializer=tf_v1.constant_initializer(weights["const_144"].asnumpy()), moving_mean_initializer=tf_v1.constant_initializer(weights["const_145"].asnumpy()), moving_variance_initializer=tf_v1.constant_initializer(weights["const_146"].asnumpy()), name="MobilenetV2_expanded_conv_14_depthwise_BatchNorm_FusedBatchNorm")
  # MobilenetV2_expanded_conv_14_depthwise_BatchNorm_FusedBatchNorm.0(get_item): <res_110> -> <res_111>
  res_111 = res_110
  # MobilenetV2_expanded_conv_14_depthwise_Relu6(clip): <res_111> -> <res_112>
  res_112 = tf_v1.clip_by_value(res_111, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_14_depthwise_Relu6")
  # gv58(msc.conv2d_bias): <res_112> -> <res_113>
  res_113 = ops.nn_ops.conv2d(res_112, weight_113, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv58")
  res_113 = ops.nn_ops.bias_add(res_113, bias_113, name="gv58_bias")
  # MobilenetV2_expanded_conv_14_add(add): <res_113,res_106> -> <res_114>
  res_114 = tf_v1.add(res_113, res_106, name="MobilenetV2_expanded_conv_14_add")
  # gv60(msc.conv2d_bias): <res_114> -> <res_115>
  res_115 = ops.nn_ops.conv2d(res_114, weight_115, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv60")
  res_115 = ops.nn_ops.bias_add(res_115, bias_115, name="gv60_bias")
  # MobilenetV2_expanded_conv_15_expand_Relu6(clip): <res_115> -> <res_116>
  res_116 = tf_v1.clip_by_value(res_115, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_15_expand_Relu6")
  # MobilenetV2_expanded_conv_15_depthwise_depthwise(nn.conv2d): <res_116> -> <res_117>
  res_117 = ops.nn_ops.depthwise_conv2d_native(res_116, weight_117, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="MobilenetV2_expanded_conv_15_depthwise_depthwise")
  # MobilenetV2_expanded_conv_15_depthwise_BatchNorm_FusedBatchNorm(nn.batch_norm): <res_117> -> <res_118>
  res_118 = tf_v1.layers.batch_normalization(res_117, scale=True, center=True, momentum=0.1, epsilon=0.001, gamma_initializer=tf_v1.constant_initializer(weights["const_148"].asnumpy()), beta_initializer=tf_v1.constant_initializer(weights["const_149"].asnumpy()), moving_mean_initializer=tf_v1.constant_initializer(weights["const_150"].asnumpy()), moving_variance_initializer=tf_v1.constant_initializer(weights["const_151"].asnumpy()), name="MobilenetV2_expanded_conv_15_depthwise_BatchNorm_FusedBatchNorm")
  # MobilenetV2_expanded_conv_15_depthwise_BatchNorm_FusedBatchNorm.0(get_item): <res_118> -> <res_119>
  res_119 = res_118
  # MobilenetV2_expanded_conv_15_depthwise_Relu6(clip): <res_119> -> <res_120>
  res_120 = tf_v1.clip_by_value(res_119, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_15_depthwise_Relu6")
  # gv62(msc.conv2d_bias): <res_120> -> <res_121>
  res_121 = ops.nn_ops.conv2d(res_120, weight_121, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv62")
  res_121 = ops.nn_ops.bias_add(res_121, bias_121, name="gv62_bias")
  # MobilenetV2_expanded_conv_15_add(add): <res_121,res_114> -> <res_122>
  res_122 = tf_v1.add(res_121, res_114, name="MobilenetV2_expanded_conv_15_add")
  # gv64(msc.conv2d_bias): <res_122> -> <res_123>
  res_123 = ops.nn_ops.conv2d(res_122, weight_123, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv64")
  res_123 = ops.nn_ops.bias_add(res_123, bias_123, name="gv64_bias")
  # MobilenetV2_expanded_conv_16_expand_Relu6(clip): <res_123> -> <res_124>
  res_124 = tf_v1.clip_by_value(res_123, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_16_expand_Relu6")
  # MobilenetV2_expanded_conv_16_depthwise_depthwise(nn.conv2d): <res_124> -> <res_125>
  res_125 = ops.nn_ops.depthwise_conv2d_native(res_124, weight_125, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="MobilenetV2_expanded_conv_16_depthwise_depthwise")
  # MobilenetV2_expanded_conv_16_depthwise_BatchNorm_FusedBatchNorm(nn.batch_norm): <res_125> -> <res_126>
  res_126 = tf_v1.layers.batch_normalization(res_125, scale=True, center=True, momentum=0.1, epsilon=0.001, gamma_initializer=tf_v1.constant_initializer(weights["const_153"].asnumpy()), beta_initializer=tf_v1.constant_initializer(weights["const_154"].asnumpy()), moving_mean_initializer=tf_v1.constant_initializer(weights["const_155"].asnumpy()), moving_variance_initializer=tf_v1.constant_initializer(weights["const_156"].asnumpy()), name="MobilenetV2_expanded_conv_16_depthwise_BatchNorm_FusedBatchNorm")
  # MobilenetV2_expanded_conv_16_depthwise_BatchNorm_FusedBatchNorm.0(get_item): <res_126> -> <res_127>
  res_127 = res_126
  # MobilenetV2_expanded_conv_16_depthwise_Relu6(clip): <res_127> -> <res_128>
  res_128 = tf_v1.clip_by_value(res_127, clip_value_min=0, clip_value_max=6, name="MobilenetV2_expanded_conv_16_depthwise_Relu6")
  # gv66(msc.conv2d_bias): <res_128> -> <res_129>
  res_129 = ops.nn_ops.conv2d(res_128, weight_129, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv66")
  res_129 = ops.nn_ops.bias_add(res_129, bias_129, name="gv66_bias")
  # gv68(msc.conv2d_bias): <res_129> -> <res_130>
  res_130 = ops.nn_ops.conv2d(res_129, weight_130, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv68")
  res_130 = ops.nn_ops.bias_add(res_130, bias_130, name="gv68_bias")
  # MobilenetV2_Conv_1_Relu6(clip): <res_130> -> <res_131>
  res_131 = tf_v1.clip_by_value(res_130, clip_value_min=0, clip_value_max=6, name="MobilenetV2_Conv_1_Relu6")
  # MobilenetV2_Logits_AvgPool(nn.avg_pool2d): <res_131> -> <res_132>
  res_132 = ops.nn_ops.pool(res_131, window_shape=[7, 7], pooling_type="AVG", dilation_rate=[1, 1], strides=[1, 1], padding="VALID", name="MobilenetV2_Logits_AvgPool")
  # gv70(msc.conv2d_bias): <res_132> -> <res_133>
  res_133 = ops.nn_ops.conv2d(res_132, weight_133, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], data_format="NHWC", padding="SAME", name="gv70")
  res_133 = ops.nn_ops.bias_add(res_133, bias_133, name="gv70_bias")
  # MobilenetV2_Logits_Squeeze(squeeze): <res_133> -> <res_134>
  res_134 = ops.array_ops.squeeze(res_133, axis=[1, 2], name="MobilenetV2_Logits_Squeeze")
  # MobilenetV2_Predictions_Reshape(reshape): <res_134> -> <res_135>
  res_135 = ops.array_ops.reshape(res_134, shape=[1, 1001], name="MobilenetV2_Predictions_Reshape")
  # MobilenetV2_Predictions_Softmax(nn.softmax): <res_135> -> <res_136>
  res_136 = tf_v1.nn.softmax(res_135, axis=-1, name="MobilenetV2_Predictions_Softmax")
  # MobilenetV2_Predictions_Reshape_1(reshape): <res_136> -> <res_137>
  res_137 = ops.array_ops.reshape(res_136, shape=[1, 1001], name="MobilenetV2_Predictions_Reshape_1")
  outputs = res_137
  return outputs


# Define the test
if __name__ == "__main__":
  # Prepare test datas
  inputs = {}
  golden = {}
  inputs["input"] = load_data("input", [1, 224, 224, 3], "float32")
  golden["MobilenetV2_Predictions_Reshape_1"] = load_data("MobilenetV2_Predictions_Reshape_1", [1, 1001], "float32")
  # Build and inference the graph
  # Load weights
  with open("main_params.bin", "rb") as f:
    params = tvm.runtime.load_param_dict(f.read())
  # Build Graph
  with tf_v1.Graph().as_default():
    res_0 = tf_v1.placeholder("float32", [1, 224, 224, 3], "input")
    outs = main(res_0, params)
    feed_dict = {}
    feed_dict[res_0] = inputs["input"]
    with tf_v1.Session() as sess:
      sess.run(ops.variables.global_variables_initializer())
      outputs = sess.run(outs, feed_dict=feed_dict)
  msc_utils.compare_arrays(golden, outputs, verbose="detail")