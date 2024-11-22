import inspect
if not hasattr(inspect, 'getargspec'):  # 修复
    inspect.getargspec = inspect.getfullargspec
import sys
from invoke import task
from d2py.tools.write import site

namespace = site('doc', target='doc/_build/html')


@task
def init(ctx):
    """初始化项目"""
    ctx.run(f"{sys.executable} -m pip install pdm")
    # 启用 {pep}`582` 模式
    ctx.run("pdm config python.use_venv false")


@task
def install(ctx, group=""):
    '''安装依赖'''
    if group:
        ctx.run(f"pdm install -G {group}")
    else:
        ctx.run("pdm install")


@task
def pdm_doc(ctx, cmd=""):
    '''文档管理'''
    if cmd:
        ctx.run(f"pdm run invoke doc.{cmd}")
    else:
        ctx.run(f"pdm run invoke doc")


@task
def tvm(ctx, root="/media/pc/data/lxw/ai/tvm/xinetzone"):
    """
    Args:
        root: TVM 根目录
    """
    ctx.run(
        f"{sys.executable} -m pdm add -d {root}")
    ctx.run(f"{sys.executable} -m pdm install")


@task
def remove(ctx, name="tvmx"):
    ctx.run(f"{sys.executable} -m pdm remove -d {name}")

namespace.add_task(init)  # 初始化项目
namespace.add_task(install)  # 安装依赖
namespace.add_task(pdm_doc)  # PDM 管理文档
namespace.add_task(tvm)  # 安装 TVM 环境
namespace.add_task(remove)  # 移除包

