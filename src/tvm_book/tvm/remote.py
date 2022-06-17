from tvm.contrib.utils import TempDirectory


class TempModule(TempDirectory):
    def export_library(self, module, filename):
        """将该 module 保存在本地临时文件夹中
        """
        path = self.relpath(filename)
        module.export_library(path)
        return path

    def upload(self, remote, module, filename):
        """将 module 上传至 remote 并返回远程模块"""
        path = self.export_library(module, filename)
        remote.upload(path)
        # remote_module
        return remote.load_module(filename)

    # def packed_func(self, remote_module, ctx):
    #     remote_fuc = remote_module["default"]
    #     remote_rt =  graph_executor.GraphModule(remote_fuc(ctx))
