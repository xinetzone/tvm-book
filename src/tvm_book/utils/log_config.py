import logging

def config_logging(
    filename,
    file_formatter="%(levelname)s|%(asctime)s|%(name)s| -> %(message)s|%(module)s.%(funcName)s@%(pathname)s:%(lineno)d",
    stream_formatter="%(levelname)s|%(asctime)s -> %(message)s"):
    """配置 logging

    Args:
        path: 日志保存路径
        file_formatter: 日志文件配置
        stream_formatter: 控制台打印配置
    """
    # 创建文件处理程序，并记录调试消息
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    # 创建日志级别更高的控制台处理程序
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO) # 或者logging.ERROR
    # 创建formatter并将其
    fh_formatter = logging.Formatter(file_formatter)
    ch_formatter = logging.Formatter(stream_formatter)
    fh.setFormatter(fh_formatter)
    ch.setFormatter(ch_formatter)
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[fh, ch]
    )