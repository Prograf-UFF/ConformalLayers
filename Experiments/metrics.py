import abc
from codecarbon import EmissionsTracker
import torch

class Metric(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def start(self, **kwargs):
        pass
    
    @abc.abstractmethod
    def stop(self, **kwargs):
        pass

    @abc.abstractmethod
    def metrics(self):
        pass


class CO2(Metric):
    def __init__(self, **kwargs) -> None:
        super(Metric, self).__init__()
        self._frequency = kwargs.get('frequency')
        self._emissions = []

    def start(self, **kwargs):
        self._tracker = EmissionsTracker(measure_power_secs = self._frequency)
        self._tracker.start()

    def stop(self, **kwargs):
        emission = self._tracker.stop()
        self._emissions.append(emission)
        return emission

    @property
    def metrics(self):
        return self._emissions


class Time(Metric):
    def __init__(self, **kwargs) -> None:
        super(Metric, self).__init__()
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._times = []
        
    def start(self, **kwargs):
        self._start.record()

    def stop(self, **kwargs):
        self._end.record()
        torch.cuda.synchronize()
        time = self._start.elapsed_time(self._end)
        self._times.append(time)
        return time
       
    @property
    def metrics(self):
        return self._times


class Memory(Metric):
    def __init__(self, **kwargs) -> None:
        super(Metric, self).__init__()
        self._measurements = []
        self._device = kwargs.get('device')
        
    def start(self, **kwargs):
        torch.cuda.reset_peak_memory_stats()

    def stop(self, **kwargs):
        measurement = torch.cuda.max_memory_allocated(device=self._device)
        self._measurements.append(measurement)
        return measurement

    @property
    def metrics(self):
        return self._measurements


class Accuracy(Metric):
    def __init__(self, **kwargs) -> None:
        super(Metric, self).__init__()
        self._measurements = []
        self._topk = kwargs.get('topk')
        
    def start(self, **kwargs):
        self._target = kwargs.get('target')

    def stop(self, **kwargs):
        self._output = kwargs.get('output')
        acc = self._compute_accuracy()
        self._measurements.append(acc)
        return acc

    def _compute_accuracy(self):
        with torch.no_grad():
            maxk = max(self._topk)
            batch_size = self._target.size(0)
            _, y_pred = self._output.topk(k=maxk, dim=1)
            y_pred = y_pred.t()
            target_reshaped = self._target.view(1, -1).expand_as(y_pred)
            correct = (y_pred == target_reshaped)
            list_topk_accs = []
            for k in self._topk:
                ind_which_topk_matched_truth = correct[:k]
                flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()
                tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)
                topk_acc = tot_correct_topk / batch_size
                list_topk_accs.append(topk_acc.item())
            return list_topk_accs


    @property
    def metrics(self):
        return self._measurements
