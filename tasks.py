import inspect
if not hasattr(inspect, 'getargspec'):  # 修复
    inspect.getargspec = inspect.getfullargspec
import sys
from invoke import task
from taolib.flows.tasks import sites

namespace = sites('doc', target='doc/_build/html')


@task
def old_doc(ctx, clean=True):
    """初始化项目"""
    with ctx.cd("tests/old"):
        if clean:
            ctx.run(f"{sys.executable} -m invoke doc.clean")
        ctx.run(f"{sys.executable} -m invoke doc")
    # # 启用 {pep}`582` 模式
    # ctx.run("pdm config python.use_venv false")

namespace.add_task(old_doc)  # 构建旧的文档


