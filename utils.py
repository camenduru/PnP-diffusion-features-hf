import datetime


def get_timestamp() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
