class Bunch(dict):
    """

    Example
    ---------
    >>> d = Bunch({'id':3, 'loc':{'x':1, 'y':2}})
    >>> d.id, d.loc.x, d["id"]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        # 支持循环嵌套
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                self.__dict__[key] = Bunch(**value)
                