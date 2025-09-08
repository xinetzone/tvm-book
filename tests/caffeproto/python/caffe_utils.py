import caffe_pb2 as pb2

def _rebuild_layers(layers: list[pb2.LayerParameter]) -> None:
    """
    处理in-place层（输入输出同名），解决名称冲突。
    优化点：使用集合加速输入名称查找（O(1)复杂度）
    """
    renamed_blobs: dict[str, str] = {}
    get_renamed = renamed_blobs.get

    # 过滤非Input层，减少迭代次数
    process_layers = [layer for layer in layers if layer.type != "Input"]
    
    for layer in process_layers:
        bottoms = layer.bottom
        tops = layer.top
        layer_name = layer.name
        bottom_set = set(bottoms)  # 转换为集合，加速in判断

        # 更新输入名称（基于历史映射）
        for idx in range(len(bottoms)):
            if (mapped := get_renamed(bottoms[idx])) is not None:
                bottoms[idx] = mapped

        # 处理in-place场景（用集合加速查找）
        for top_idx, top_name in enumerate(tops):
            if top_name in bottom_set:  # O(1)查找，替代原列表in操作
                # 多输出加索引后缀，确保唯一性
                normalized_top = f"{layer_name}_{top_idx}" if len(tops) > 1 else layer_name
                renamed_blobs[top_name] = normalized_top
                tops[top_idx] = normalized_top


def unity_inputs(predict_net: pb2.NetParameter) -> pb2.NetParameter:
    """
    转换旧输入定义为显式Input层，彻底清除input/input_shape/input_dim。
    优化点：精简条件判断，使用海象运算符减少变量赋值
    """
    try:
        # 检查是否已有Input层（提前返回，避免无效处理）
        if any(layer.type == "Input" for layer in predict_net.layer):
            predict_net.ClearField("input")
            predict_net.ClearField("input_shape")
            predict_net.ClearField("input_dim")
            return predict_net

        # 处理旧输入定义（仅当input字段非空时）
        if predict_net.input:
            input_names = list(predict_net.input)
            # 解析形状（用三元表达式精简逻辑）
            shapes = ([list(sp.dim) for sp in predict_net.input_shape] 
                      if predict_net.input_shape 
                      else [list(predict_net.input_dim)])

            # 校验名称与形状数量
            if len(input_names) != len(shapes):
                raise ValueError(f"输入名称({len(input_names)})与形状({len(shapes)})不匹配")

            # 生成并插入Input层
            input_layers = [
                pb2.LayerParameter(
                    name=name,
                    type="Input",
                    top=[name],
                    input_param=pb2.InputParameter(shape=[pb2.BlobShape(dim=shape)])
                )
                for idx, (name, shape) in enumerate(zip(input_names, shapes))
                if all(isinstance(dim, int) and dim > 0 for dim in shape)  # 形状校验整合到生成式
            ]

            # 检查是否有无效形状（生成式会过滤，导致数量不匹配）
            if len(input_layers) != len(input_names):
                raise TypeError("输入形状包含无效维度（需为正整数）")

            # 插入Input层（保持原始顺序）
            for layer in reversed(input_layers):
                predict_net.layer.insert(0, layer)

        # 强制清除旧属性（无论是否处理过旧输入）
        predict_net.ClearField("input")
        predict_net.ClearField("input_shape")
        predict_net.ClearField("input_dim")

        return predict_net

    except Exception as e:
        # 异常兜底清除
        for field in ["input", "input_shape", "input_dim"]:
            try:
                predict_net.ClearField(field)
            except:
                pass
        raise RuntimeError(f"输入层转换失败：{str(e)}") from e


def convert_num_to_name(predict_net: pb2.NetParameter) -> pb2.NetParameter:
    """
    统一输入输出名称为层名（支持多输出）。
    优化点：合并循环逻辑，减少属性访问
    """
    blob_mapping: dict[str, str] = {}
    get_mapped = blob_mapping.get

    for layer in predict_net.layer:
        layer_name = layer.name
        bottoms, tops = layer.bottom, layer.top  # 合并赋值

        # 处理Input层输出
        if layer.type == "Input":
            for top_idx, top_name in enumerate(tops):
                normalized = f"{layer_name}_{top_idx}" if len(tops) > 1 else layer_name
                blob_mapping[top_name] = normalized
                tops[top_idx] = normalized
            continue

        # 更新输入名称
        for idx in range(len(bottoms)):
            if (mapped := get_mapped(bottoms[idx])) is not None:
                bottoms[idx] = mapped

        # 标准化输出名称
        for idx, top_name in enumerate(tops):
            normalized = f"{layer_name}_{idx}" if len(tops) > 1 else layer_name
            blob_mapping[top_name] = normalized
            tops[idx] = normalized

    return predict_net


def unity_struct(predict_net: pb2.NetParameter) -> pb2.NetParameter:
    """
    整合模型结构标准化流程，确保输出模型无旧输入属性残留。
    优化点：明确步骤依赖，添加类型提示
    """
    try:
        # 步骤1：处理输入层，清除旧属性
        predict_net = unity_inputs(predict_net)
        # 步骤2：处理in-place层冲突
        _rebuild_layers(predict_net.layer)
        # 步骤3：统一名称规范
        predict_net = convert_num_to_name(predict_net)
        # 最终确认清除（双重保障）
        for field in ["input", "input_shape", "input_dim"]:
            predict_net.ClearField(field)
        return predict_net
    except Exception as e:
        raise RuntimeError(f"模型标准化失败：{str(e)}") from e

