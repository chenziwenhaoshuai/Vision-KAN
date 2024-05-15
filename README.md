# Vision-KAN
We are experimenting with the possibility of KAN replacing MLP in Vision Transformer, this project may be delayed for a long time due to GPU resource constraints, if there are any new developments, we will show them here!
|**ImageNet 1k**|
|-------|-------|-------|-------|-------|
| model | date | epoch | top1 | top5 | 
|-------|-------|-------|-------|-------|
| Vision-KAN | 2024.5.15 | 16 | 32.65 | 57.03 |

# News
## 5.7.2024
We released our current Vision KAN code, we used efficient KAN to simply replace the MLP layer in the Transformer block and are pre-training the Tiny model on ImageNet 1k, subsequent results will be updated in the table.
## 5.14.2024
The model has started to converge, we use [192, 20, 192] as input, hidden, and output dimensions, and we reshape the input dimensions in order to fit the processing dimensions of KAN.
## 5.15.2024
we change ekan to faster kan to speed up to 2x in training process, and change basemodel from Deit iii to Deit, so that we can use pretrained model for most layers except kan layer
# Architecture
We used [DeiT](https://github.com/facebookresearch/deit) as a baseline for Vision KAN development, thanks to Meta and MIT for the amazing work!
