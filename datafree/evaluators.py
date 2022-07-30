from tqdm import tqdm
import torch.nn.functional as F 
import torch
from . import metrics

class Evaluator(object):
    def __init__(self, metric, dataloader):
        self.dataloader = dataloader
        self.metric = metric
        # self.logit_kl_metric = logit_kl_metric

    def eval(self, model, device=None, progress=False):
        self.metric.reset()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate( tqdm(self.dataloader, disable=not progress) ):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model( inputs )
                self.metric.update(outputs, targets)
                # if model_G is not None:
                #     z = torch.randn(input.size(0), model_G.nz).to(device)
                #     x = normalizer(model_G(z))
                #     y_logit = model(x)
                #     self.logit_kl_metric.update(torch.log_softmax(y_logit, 1), torch.softmax(outputs, 1))
        return self.metric.get_results()
    
    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

class AgreeEvaluator(object):
    def __init__(self, metric, dataloader):
        self.dataloader = dataloader
        self.metric = metric

    def eval(self, model_t, model_s, device=None, progress=False):
        self.metric.reset()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(tqdm(self.dataloader, disable=not progress)):
                inputs = inputs.to(device)
                logit_t = model_t(inputs)
                logit_s = model_s(inputs)
                self.metric.update(logit_s, logit_t)
        return self.metric.get_results()
    
    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)
        
class YKLEvaluator(object):
    def __init__(self, metric, dataloader, logit_kl_metric=None):
        self.dataloader = dataloader
        self.metric = metric
        self.logit_kl_metric = logit_kl_metric

    def eval(self, model, device=None, progress=False, G_list=None, normalizer=None):
        self.metric.reset()
        # print(device)
        # print('************')
        # print(device)
        # exit(-1)
        # if G_list is not None:
        #     z = torch.randn(self.dataloader.batch_size, G_list[0].nz).to(device)
        with torch.no_grad():
            for i, (inputs, targets) in enumerate( tqdm(self.dataloader, disable=not progress) ):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model( inputs )
                self.metric.update(outputs, targets)
                if G_list is not None:
                    z = torch.randn(inputs.size(0), G_list[0].nz).to(device)
                    for l, model_G in enumerate(G_list):
                        if l > 0:
                            func = F.relu
                        else:
                            func = normalizer
                        x = func(model_G(z, l=l))
                        y_logit = model(x, l=l)
                        # print(y_logit.shape, outputs.shape, l)
                        self.logit_kl_metric.update(y_logit, outputs)
        return {'normal': self.metric.get_results(), 'y_kl': self.logit_kl_metric.get_results()}
    
    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

class AdvEvaluator(object):
    def __init__(self, metric, dataloader, adversary):
        self.dataloader = dataloader
        self.metric = metric
        self.adversary = adversary

    def eval(self, model, device=None, progress=False):
        self.metric.reset()
        for i, (inputs, targets) in enumerate( tqdm(self.dataloader, disable=not progress) ):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = self.adversary.perturb(inputs, targets)
            with torch.no_grad():
                outputs = model( inputs )
                self.metric.update(outputs, targets)
        return self.metric.get_results()
    
    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

def classification_evaluator(dataloader):
    metric = metrics.MetricCompose({
        'Acc': metrics.TopkAccuracy(),
        'Loss': metrics.RunningLoss(torch.nn.CrossEntropyLoss(reduction='sum')),
        
    })
    
    return Evaluator( metric, dataloader=dataloader)

def ykl_classification_evaluator(dataloader, L=1):
    metric = metrics.MetricCompose({
        'Acc': metrics.TopkAccuracy(),
        'Loss': metrics.RunningLoss(torch.nn.CrossEntropyLoss(reduction='sum')),
        
    })
    kl_dict = {}
    for l in range(L):
        kl_dict['kl_at_{}'.format(l)] = metrics.RunningLoss(torch.nn.KLDivLoss(reduction='sum'))
    logit_kl_metric = metrics.MetricCompose(kl_dict)
    # logit_kl_metric = metrics.RunningLoss(torch.nn.KLDivLoss(reduction='sum'))
    return YKLEvaluator(metric, dataloader=dataloader, logit_kl_metric=logit_kl_metric)

def advarsarial_classification_evaluator(dataloader, adversary):
    metric = metrics.MetricCompose({
        'Acc': metrics.TopkAccuracy(),
        'Loss': metrics.RunningLoss(torch.nn.CrossEntropyLoss(reduction='sum'))
    })
    return AdvEvaluator( metric, dataloader=dataloader, adversary=adversary)


def segmentation_evaluator(dataloader, num_classes, ignore_idx=255):
    cm = metrics.ConfusionMatrix(num_classes, ignore_idx=ignore_idx)
    metric = metrics.MetricCompose({
        'mIoU': metrics.mIoU(cm),
        'Acc': metrics.Accuracy(),
        'Loss': metrics.RunningLoss(torch.nn.CrossEntropyLoss(reduction='sum'))
    })
    return Evaluator( metric, dataloader=dataloader)

# Read paper, make conclusion, and design experiment

def prediction_agreement_evaluator(dataloader):
    metric = metrics.MetricCompose({
        'agreement': metrics.Accuracy(),
        'prob_loyalty': metrics.ProbLoyalty(),
    })
    return AgreeEvaluator(metric, dataloader=dataloader)

def difficulty_evaluator(dataloader):
    metric = metrics.RunningLoss(torch.nn.KLDivLoss(reduction='sum'))
    return Evaluator(metric, dataloader=dataloader)

# def lotalty_evaluator(dataloader):
#     pass

# def prob_lotalty_evaluator(dataloader):
#     pass