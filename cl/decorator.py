import threading
from typing import Any, Type


def sync(lock: threading.Lock):
    def function(f):
        def wrapper(*args, **kargs):
            lock.acquire()
            try:
                return f(*args, **kargs)
            finally:
                lock.release()
        return wrapper
    return function


def singleton(cls: Type[Any]):
    instances = {}
    def instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return instance
