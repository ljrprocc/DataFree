# DataFree Knowledge Distillation By Curriculum Learning

Forked by a benchmark of data-free knowledge distillation from paper "How to Teach: Learning Data-Free Knowledge Distillation From Curriculum".
Forked by [CMI](https://arxiv.org/abs/2105.08584).

## Installation
We use Pytorch for implementation. Please install the following requirement packages
```
pip install -r requirements.txt
```

## Our method
Based on curriculum learning and self-paced learning. Our method is called **CuDFKD**. The result can be rephrased by scripts `scripts/cudfkd/`. For example, when distill ResNet18 from ResNet34 at the benchmark CIFAR10, please run the following script

```
bash scripts/cudfkd/cudfkd_cifar10_resnet34_resnet18.sh
```

The implementation is in `datafree/synthesis/cudfkd.py`.

## Result on CIFAR10
| Teacher    | Res34 | vgg11 | wrn-402 | wrn-402 | wrn-402 |
|------------|-------|-------|---------|---------|---------|
| Student    | Res18 | Res18 | wrn-401 | wrn-162 | wrn-161 |
| T. Scratch | 95.70 | 92.25 | 94.87   | 94.87   | 94.87   |
| S. Scratch | 95.20 | 95.20 | 93.94   | 93.95   | 91.12   |
| DAFL       | 92.22 | 81.10 | 81.33   | 81.55   | 72.15   |
| ZSKT       | 93.32 | 89.46 | 86.07   | 89.66   | 83.74   |
| ADI        | 93.26 | 90.36 | 86.85   | 89.72   | 83.01   |
| DFQ        | 94.61 | 90.84 | 91.69   | 92.01   | 86.14   |
| CMI        | 94.84 | 91.13 | 92.78   | 92.52   | **90.01**   |
| PRE-DFKD   | 94.10 | N\A   | N\A     | N\A     | N\A     |
| CuDFKD     | **95.28** | **91.61** | **93.18**   | **92.98**   | 88.77   |

GPU and time usage when using a simple NVIDIA 3090 TI with batch size 256.

|          | $\mu$ | $\sigma^2$ | best | Memory  | Time  |
|----------|-------|------------|------|-------|-------|
| DAFL     | 62.6  | 17.1       | 92.0 | **6.45G** | 6.10h |
| DFAD     | 86.1  | 12.3       | 93.3 | -     | -     |
| ADI      | 87.2  | 13.9       | 93.3 | 7.85G | 25.2h |
| CMI      | 82.4  | 16.6       | 94.8 | 12.5G | 13.3h |
| MB-DFKD  | 83.3  | 16.4       | 92.4 | -     | -     |
| PRE-DFKD | 87.4  | 10.3       | 94.1 | -     | -     |
| CuDFKD   | **94.1**  | **2.88**       | **95.0** | 6.84G | **5.48h** |

## Result on CIFAR100

| Teacher    | Res34 | vgg11 | wrn-402 | wrn-402 |
|------------|-------|-------|---------|---------|
| Student    | Res18 | Res18 | wrn-401 | wrn-162 |
| T. Scratch | 78.05 | 71.32 | 75.83   | 75.83   |
| S. Scratch | 77.10 | 77.10 | 72.19   | 73.56   |
| DAFL       | 74.47 | 57.29 | 34.66   | 40.00   |
| ZSKT       | 67.74 | 34.72 | 29.73   | 28.44   |
| ADI        | 61.32 | 54.13 | 61.33   | 61.34   |
| DFQ        | 77.01 | 68.32 | 61.92   | 59.01   |
| CMI        | 77.02 | 70.56 | **68.88**   | **68.57**   |
| PRE-DFKD   | **77.04** | N\A   | N\A     | N\A     |
| CuDFKD     | 75.87 | **71.22** | 66.43   | 65.94   |

|          | $\mu$ | $\sigma^2$ | best | Mem   | Time  |
|----------|-------|------------|------|-------|-------|
| DAFL     | 52.5  | 12.8       | 74.5 | **6.45G** | **7.09h** |
| DFAD     | 54.9  | 12.9       | 67.7 | -     | -     |
| ADI      | 51.3  | 18.2       | 61.3 | 7.85G | 30.4h |
| CMI      | 55.2  | 24.1       | 77.0 | 12.5G | 22.3h |
| MB-DFKD  | 64.4  | 18.3       | 75.4 | -     | -     |
| PRE-DFKD | 70.2  | 11.1       | **77.1** | -     | -     |
| CuDFKD   | **71.7**  | **4.37**       | 75.9 | 6.84G | 7.50h |


## Other visualization results
Please refer to the supplementary material pdf.

## Reference

* ZSKT: [Zero-shot Knowledge Transfer via Adversarial Belief Matching](https://arxiv.org/abs/1905.09768)
* DAFL: [Data-Free Learning of Student Networks](https://arxiv.org/abs/1904.01186)
* DeepInv: [Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion](https://arxiv.org/abs/1912.08795)
* DFQ: [Data-Free Network Quantization With Adversarial Knowledge Distillation](https://arxiv.org/abs/2005.04136)
* CMI: [Contrastive Model Inversion for Data-Free Knowledge Distillation](https://arxiv.org/abs/2105.08584)