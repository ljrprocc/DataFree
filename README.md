# DataFree Knowledge Distillation By Curriculum Learning

Forked by a benchmark of data-free knowledge distillation from paper "How to Teach: Learning Data-Free Knowledge Distillation From Curriculum".
Forked by [CMI](https://arxiv.org/abs/2105.08584).

## Installation
We use Pytorch for implementation. Please install the following requirement packages
```
pip install -r requirements.txt
```

## Our method
Based on curriculum learning and self-paced learning. Our method is called **CuDFKD**. The result can be rephrased by scripts `scripts/probkd/`. For example, when distill ResNet18 from ResNet34 at the benchmark CIFAR10, please run the following script

```
bash scripts/probkd/probkd_cifar10_resnet34_resnet18.sh
```

The implementation is in `datafree/synthesis/probkd.py`.

## Result on CIFAR10
Please refer to paper.

## Result on CIFAR100
Please refer to paper.


## Other visual results
Please refer to the supplementary material pdf.

## Reference

* ZSKT: [Zero-shot Knowledge Transfer via Adversarial Belief Matching](https://arxiv.org/abs/1905.09768)
* DAFL: [Data-Free Learning of Student Networks](https://arxiv.org/abs/1904.01186)
* DeepInv: [Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion](https://arxiv.org/abs/1912.08795)
* DFQ: [Data-Free Network Quantization With Adversarial Knowledge Distillation](https://arxiv.org/abs/2005.04136)
* CMI: [Contrastive Model Inversion for Data-Free Knowledge Distillation](https://arxiv.org/abs/2105.08584)