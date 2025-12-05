import pytest


def test_vta_not_linked_error():
    import flexloopy
    import flexloopy.vta_ffi as vta
    # 当库存在但未链接 VTA 时应抛出一致错误；若库不存在，跳过
    try:
        with pytest.raises(Exception):
            vta.uop_loop_end()
    except FileNotFoundError:
        pytest.skip("shared library not built")