def sync(lock):
    def function(f):
        def wrapper(*args, **kargs):
            lock.acquire()
            try:
                return f(*args, **kargs)
            finally:
                lock.release ()
        return wrapper
    return function


def singleton(cls):
    instances = {}
    def instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return instance
