"""库信息"""
import os
import sys

def split_env_var(env_var: str, split: str)-> list[str]:
    """将环境变量字符串拆分。

    Args:
        env_var: 环境变量的名称
        split: 需要拆分的环境变量字符串。
    """
    if os.environ.get(env_var, None):
        return [p.strip() for p in os.environ[env_var].split(split)]
    return []