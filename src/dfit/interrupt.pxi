cdef object threading
import threading

cdef object sys
import sys


def _timed_run(func, tuple args=(), dict kwargs={}, timeout=30, default=None):
    """This function will spawn a thread and run the given function
    using the args, kwargs and return the given default value if the
    timeout is exceeded.

    http://stackoverflow.com/questions/492519/timeout-on-a-python-function-call
    """

    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = default
            self.exc_info = (None, None, None)

        def run(self):
            try:
                self.result = func(args, **kwargs)
            except Exception as err:
                self.exc_info = sys.exc_info()

        def suicide(self):
            raise RuntimeError(f"Timeout (taking more than {timeout} sec)")

    it = InterruptableThread()
    it.start()
    it.join(timeout)

    if it.exc_info[0] is not None:
        a, b, c = it.exc_info
        raise Exception(a, b, c)  # communicate that to caller

    if it.isAlive():
        it.suicide()
        raise RuntimeError
    else:
        return it.result