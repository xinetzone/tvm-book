class Bunch(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self # 这意味着 Bunch 类的实例将具有与字典相同的行为，可以使用点符号访问和修改其键值对
        self._convert_nested_dicts()

    def _convert_nested_dicts(self):
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                self.__dict__[k] = Bunch(**v)  # 将字典转换为 Bunch 对象
            elif isinstance(v, Bunch):
                v._convert_nested_dicts()  # 递归处理嵌套的 Bunch 对象
                     
    def merge(self, other):
        """提供递归合并功能"""
        other = Bunch(**other)
        for k, v in other.items():
            if k not in self:
                self[k] = other[k]
            else:
                if not isinstance(self[k], dict) and not isinstance(v, dict):
                    self[k] = v
                elif isinstance(self[k], dict) and isinstance(v, dict):
                    self[k].update(v)
                else:
                    raise TypeError(f"{other}不支持合并")