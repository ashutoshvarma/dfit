cdef object sys
import sys

cdef object traceback
import traceback

cdef object MpTimeoutError
from multiprocessing import TimeoutError as MpTimeoutError

cdef object Queue_Empty
from queue import Empty as Queue_Empty

cdef object Queue
from queue import Queue

cdef object start_new_thread
from _thread import start_new_thread


class TimeoutError(Exception):
    pass


cdef int async_raise(unsigned long tid, object exception=Exception) except -1:
    """
    Raise an Exception in the Thread with id `tid`. Perform cleanup if
    needed.
    Based on Killable Threads By Tomer Filiba
    from http://tomerfiliba.com/recipes/Thread2/
    license: public domain.
    """
    res = cpython.PyThreadState_SetAsyncExc(tid, <PyObject*>exception)
    if res == 0:
        raise ValueError('Invalid thread id.')
    elif res != 1:
        # if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect
        cpython.PyThreadState_SetAsyncExc(tid, <PyObject*>NULL)
        raise SystemError('PyThreadState_SetAsyncExc failed.')


def interrupt_func(func, tuple args=(), dict kwargs={}, timeout=30, q=None):
    """
    Threads-based interruptible runner, but is not reliable and works
    only if everything is pickable.
    """
    cdef:
        unsigned long tid

    # We run `func` in a thread and block on a queue until timeout
    if not q:
        q = Queue()

    def runner():
        try:
            _res = func(*args, **kwargs)
            q.put((None, _res))
        except TimeoutError:
            # rasied by async_rasie to kill the orphan threads
            pass
        except Exception as ex:
            q.put((ex, None))

    tid = start_new_thread(runner, ())

    try:
        resp = q.get(timeout=timeout)
        if resp[0]:
            raise resp[0]
        return resp[1]
    except (Queue_Empty, MpTimeoutError):
        raise TimeoutError(f"Timeout (taking more than {timeout} sec)")
    finally:
        try:
            async_raise(tid, TimeoutError)
        except (SystemExit, ValueError):
            pass


def fake_interrupt_func(func, tuple args=(), dict kwargs={}, timeout=30, q=None):
    """
    For debug purpose
    """
    return func(*args, **kwargs)
