"""废弃"""

import json


def update_vta_config(tvm_root, **kw):
    """修改 :file:`vta_config.json`

    Args:
        tvm_root: 即 TVM 项目的根目录
        kw: 用于配置硬件参数的字典
    
    Examples:
        >>> update_vta_config(tvm_root, **{'TARGET': "sim"})
    """
    vat_config = f"{tvm_root}/3rdparty/vta-hw/config/vta_config.json"
    with open(vat_config, encoding="utf-8") as fp:
        config = json.load(fp)

    config.update(kw)
    with open(vat_config, "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=4)
