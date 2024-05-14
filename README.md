# Vision-KAN
We are experimenting with the possibility of KAN replacing MLP in Vision Transformer, this project may be delayed for a long time due to GPU resource constraints, if there are any new developments, we will show them here!
| model | date | epoch | top1 | top5 | 
|-------|-------|-------|-------|-------|
| Vision-KAN | 2024.5.14 | 21 | 24.612 | 47.214 |

# News
## 5.7.2024
We released our current Vision KAN code, we used efficient KAN to simply replace the MLP layer in the Transformer block and are pre-training the Tiny model on ImageNet 1k, subsequent results will be updated in the table.
## 5.14.2024
The model has started to converge, we use 192,20,192 as input, hidden, and output dimensions, and we reshape the input dimensions in order to fit the processing dimensions of KAN.
# Architecture
We used [DeiT](https://github.com/facebookresearch/deit) as a baseline for Vision KAN development, thanks to Meta and MIT for the amazing work!
