from tvm.contrib.utils import TempDirectory


class TempModule(TempDirectory):
    """Create temp dir which deletes the contents when exit.

    Parameters
    ----------
    custom_path : str, optional
        Manually specify the exact temp dir path

    keep_for_debug : bool
        Keep temp directory for debugging purposes
    Returns
    -------
    temp : TempDirectory
        The temp directory object
    """
    def export_library(self, module, filename, *args, **kwargs):
        """将该 module 保存在本地临时文件夹中
        """
        path = self.relpath(filename)
        module.export_library(path, *args, **kwargs)
        return path

    def upload(self, remote, module, filename):
        """将 module 上传至 remote 并返回远程模块"""
        path = self.export_library(module, filename)
        remote.upload(path)
        # remote_module
        return remote.load_module(filename)
        