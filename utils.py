import time, warnings
from typing import Any, Callable, Dict, Optional


class Stopwatch(object):
    def __init__(self, message: Optional[str]=None, extra_map: Dict[str, Any]=dict(), *, action: Optional[Callable[['Stopwatch'], None]]=None) -> None:
        if message is None:
            message = 'Elapsed time: {et_str}.'
        if action is None:
            action = lambda sw, extra_map: print(sw.message.format_map({'start': sw.start, 'stop': sw.stop, 'et': sw.et, 'et_str': self._format_time(sw.et), **extra_map}))
        self._action = action
        self._extra_map = extra_map
        self._message = message
        self._start = None
        self._stop = None

    def __enter__(self) -> 'Stopwatch':
        self._start = time.time()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        if exception_type is None:
            self._stop = time.time()
            self._action(self, self._extra_map)
        else:
            self._stop = None

    def _format_time(self, seconds: float) -> str:
        days = int(seconds / 3600/24)
        seconds = seconds - days*3600*24
        hours = int(seconds / 3600)
        seconds = seconds - hours*3600
        minutes = int(seconds / 60)
        seconds = seconds - minutes*60
        secondsf = int(seconds)
        seconds = seconds - secondsf
        millis = int(seconds*1000)

        f = ''
        i = 1
        if days > 0:
            f += str(days) + 'D'
            i += 1
        if hours > 0 and i <= 2:
            f += str(hours) + 'h'
            i += 1
        if minutes > 0 and i <= 2:
            f += str(minutes) + 'm'
            i += 1
        if secondsf > 0 and i <= 2:
            f += str(secondsf) + 's'
            i += 1
        if millis > 0 and i <= 2:
            f += str(millis) + 'ms'
            i += 1
        if f == '':
            f = '0ms'
        return f

    @property
    def action(self) -> Callable[['Stopwatch'], None]:
        return self._action

    @property
    def et(self) -> float:
        if self.start is None or self.stop is None:
            warnings.warn('Stopwatch.et called before leaving the "with" statement.', RuntimeWarning, stacklevel=2)
            return float('NaN')
        return (self.stop - self.start)

    @property
    def extra_map(self) -> Dict[str, Any]:
        return self._extra_map
    
    @property
    def message(self) -> str:
        return self._message

    @property
    def start(self) -> float:
        return self._start

    @property
    def stop(self) -> float:
        return self._stop
