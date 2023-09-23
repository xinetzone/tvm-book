def assert_shapes_match(tru, est):
    """验证形状是否相等"""
    if tru.shape != est.shape:
        raise AssertionError(f"输出形状 {tru.shape} 和 {est.shape} 不匹配")