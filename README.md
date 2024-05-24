# Vision-KAN
We are experimenting with the possibility of [KAN](https://github.com/KindXiaoming/pykan) replacing MLP in Vision Transformer, this project may be delayed for a long time due to GPU resource constraints, if there are any new developments, we will show them here!
| Dataset | MLP hidden dim | model | date | epoch | top1 | top5 | 
|-------|-------|-------|-------|-------|-------|-------|
| ImageNet 1k | 768 | DeiT-tiny(baseline) | - | 300 | 72.2 | 91.1 |

| Dataset | KAN hidden dim | model | date | epoch | top1 | top5 | 
|-------|-------|-------|-------|-------|-------|-------|
| ImageNet 1k | 20 | Vision-KAN | 2024.5.16 | 37(stop) | 36.34 | 61.48 |
| ImageNet 1k | 192 | Vision-KAN | 2024.5.24 | 305(training) | 64.86 | 86.06 |

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
# Architecture
We used [DeiT](https://github.com/facebookresearch/deit) as a baseline for Vision KAN development, thanks to Meta and MIT for the amazing work!
# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=chenziwenhaoshuai/Vision-KAN&type=Date)](https://star-history.com/#chenziwenhaoshuai/Vision-KAN&Date)
