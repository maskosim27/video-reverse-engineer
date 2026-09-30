"""Job queue: in-process ThreadPoolExecutor.
ponytail: global pool, 2 workers. Redis/Celery only if this ever needs scale
or surviving restarts — for a local single-user app it doesn't."""
from concurrent.futures import ThreadPoolExecutor

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vre")


def submit(fn, *args, **kwargs):
    return _pool.submit(fn, *args, **kwargs)
