import numpy as np
import torch
from .stream_metrics import Metric
from typing import Callable
from scipy.spatial import distance

__all__=['ProbLoyalty']

class ProbLoyalty(Metric):
    def __init__(self):
        self.reset()

    @torch.no_grad()
    def update(self, outputs, targets):
        outputs = torch.softmax(outputs, 1) + 1e-8
        targets = torch.softmax(targets, 1) + 1e-8
        outputs = outputs.detach().cpu().numpy()
        targets = targets.cpu().numpy()
        assert len(outputs.shape) == 2
        assert len(targets.shape) == 2
        js = distance.jensenshannon(outputs.T, targets.T)
        self._pl += np.nansum(1 - np.sqrt(js))
        self._cnt += np.sum(~np.isnan(js))

    def get_results(self):
        return self._pl / self._cnt
    
    def reset(self):
        self._pl = self._cnt = 0.0


class TopkAccuracy(Metric):
    def __init__(self, topk=(1, 5)):
        self._topk = topk
        self.reset()
    
    @torch.no_grad()
    def update(self, outputs, targets):
        for k in self._topk:
            _, topk_outputs = outputs.topk(k, dim=1, largest=True, sorted=True)
            correct = topk_outputs.eq( targets.view(-1, 1).expand_as(topk_outputs) )
            self._correct[k] += correct[:, :k].view(-1).float().sum(0).item()
        self._cnt += len(targets)

    def get_results(self):
        return tuple( self._correct[k] / self._cnt * 100. for k in self._topk )

    def reset(self):
        self._correct = {k: 0 for k in self._topk}
        self._cnt = 0.0