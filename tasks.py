import inspect
if not hasattr(inspect, 'getargspec'):  # 修复
    inspect.getargspec = inspect.getfullargspec
import sys
from invoke import task
from taolib.doc import sites

namespace = sites('doc', target='doc/_build/html')
