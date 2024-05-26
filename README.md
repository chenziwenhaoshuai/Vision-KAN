# Vision-KAN
We are experimenting with the possibility of [KAN](https://github.com/KindXiaoming/pykan) replacing MLP in Vision Transformer, this project may be delayed for a long time due to GPU resource constraints, if there are any new developments, we will show them here!
| Dataset | MLP hidden dim | model | date | epoch | top1 | top5 | 
|-------|-------|-------|-------|-------|-------|-------|
| ImageNet 1k | 768 | DeiT-tiny(baseline) | - | 300 | 72.2 | 91.1 |
| CIFAR-100 | 192 | DeiT-tiny(baseline) | 2024.5.25 | 300(stop) | 84.94 | 96.53 |
| CIFAR-100 | 384 | DeiT-small(baseline) | 2024.5.25 | 300(stop) | 86.49 | 96.17 |
| CIFAR-100 | 768 | DeiT-base(baseline) | 2024.5.25 | 300(stop) | 86.54 | 96.16 |

| Dataset | KAN hidden dim | model | date | epoch | top1 | top5 | Checkpoint |
|-------|-------|-------|-------|-------|-------|-------|-------|
| ImageNet 1k | 20 | Vision-KAN | 2024.5.16 | 37(stop) | 36.34 | 61.48 | - |
| ImageNet 1k | 192 | Vision-KAN | 2024.5.25 | 346(stop) | 64.87 | 86.14 |[Checkpoint](https://pan.baidu.com/s/117ox7oh6zzXLwPMmQ6od1Q?pwd=y1vw) |
| ImageNet 1k | 768 | Vision-KAN | 2024.5.26 | 17(training) | 48.94 | 73.68 | - |
| CIFAR-100 | 192 | Vision-KAN | 2024.5.25 | 300(stop) | 73.17 | 93.307 | [Checkpoint](https://drive.google.com/drive/folders/19WPq6bZ9NgX-WxD7qXSTKiHc5D6P8jQP?usp=sharing) |
| CIFAR-100 | 384 | Vision-KAN | 2024.5.25 | 300(stop) | 78.69 | 94.73 | - |
| CIFAR-100 | 768 | Vision-KAN | 2024.5.25 | 191(training) ETA-12h (300 epochs) | 76.96 | 94.36 | - |

# News
## 5.7.2024
We released our current Vision KAN code, we used efficient KAN to simply replace the MLP layer in the Transformer block and are pre-training the Tiny model on ImageNet 1k, subsequent results will be updated in the table.
## 5.14.2024
The model has started to converge, we use [192, 20, 192] as input, hidden, and output dimensions, and we reshape the input dimensions in order to fit the processing dimensions of KAN.
## 5.15.2024
we change [efficient kan](https://github.com/Blealtan/efficient-kan) to [faster kan](https://github.com/AthanasiosDelis/faster-kan) to speed up to 2x in training process, and change base model from Deit iii to Deit, so that we can use pre-trained model for most layers except kan layer
## 5.16.2024
The convergence of the model seems to be entering a bottleneck, and I'm guessing that kan's hidden layer setting of 20 is too small, so I'm going to adjust the hidden layer to 192 if it doesn't converge after a few more rounds of running.
## 5.22.2024
Fix Timm version dependency bugs and remove extraneous code.
## 5.24.2024
The decline in losses is starting to slow down and it looks like it's getting close to the final result.
## 5.25.2024
The model with 192 hidden layers is close to convergence and we will next try a larger KAN hidden layer, the same as the MLP.
We release the best checkpoint of VisionKAN with 192 hidden dim.
# Architecture
We used [DeiT](https://github.com/facebookresearch/deit) as a baseline for Vision KAN development, thanks to Meta and MIT for the amazing work!
# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=chenziwenhaoshuai/Vision-KAN&type=Date)](https://star-history.com/#chenziwenhaoshuai/Vision-KAN&Date)
