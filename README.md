# Vision-KAN 🚀

Welcome to **Vision-KAN**! We are exploring the exciting possibility of [KAN](https://github.com/KindXiaoming/pykan) replacing MLP in Vision Transformer. Due to GPU resource constraints, this project may experience delays, but we'll keep you updated with any new developments here! 📅✨

## Installation 🛠️

To install this package, simply run:

```bash
pip install VisionKAN
```

## Minimal Example 💡

Here's a quick example to get you started:

```python
from VisionKAN import create_model, train_one_epoch, evaluate

KAN_model = create_model(
    model_name='deit_tiny_patch16_224_KAN',
    pretrained=False,
    hdim_kan=192,
    num_classes=100,
    drop_rate=0.0,
    drop_path_rate=0.05,
    img_size=224,
    batch_size=144
)
```

## Performance Overview 📊

### Baseline Models

| Dataset | MLP Hidden Dim | Model               | Date | Epoch | Top-1 | Top-5 | Checkpoint |
|---------|----------------|---------------------|------|-------|-------|-------|------------|
| ImageNet 1k | 768          | DeiT-tiny (baseline) | -    | 300   | 72.2  | 91.1  | -          |
| CIFAR-100   | 192          | DeiT-tiny (baseline) | 2024.5.25 | 300(stop) | 84.94 | 96.53 | [Checkpoint](https://drive.google.com/drive/folders/1hPrnfI5CKMgwM6lgSrFUwvMQYsjtjg3A?usp=drive_link) |
| CIFAR-100   | 384          | DeiT-small (baseline) | 2024.5.25 | 300(stop) | 86.49 | 96.17 | [Checkpoint](https://drive.google.com/drive/folders/1ZSl2ojZUQRkIsZzJ0w5rahOTAv4IiZCt?usp=drive_link) |
| CIFAR-100   | 768          | DeiT-base (baseline)  | 2024.5.25 | 300(stop) | 86.54 | 96.16 | [Checkpoint](https://drive.google.com/drive/folders/14kLdJDy11zv_mC35JvbcPCdoXvrHspNK?usp=sharing) |

### Vision-KAN Models

| Dataset | KAN Hidden Dim | Model     | Date     | Epoch     | Top-1 | Top-5 | Checkpoint |
|---------|----------------|-----------|----------|-----------|-------|-------|------------|
| ImageNet 1k | 20         | Vision-KAN | 2024.5.16 | 37(stop)  | 36.34 | 61.48 | -          |
| ImageNet 1k | 192        | Vision-KAN | 2024.5.25 | 346(stop) | 64.87 | 86.14 | [Checkpoint](https://pan.baidu.com/s/117ox7oh6zzXLwPMmQ6od1Q?pwd=y1vw) |
| ImageNet 1k | 768        | Vision-KAN | 2024.6.2  | 154(training) | 62.90 | 85.03 | -          |
| CIFAR-100   | 192        | Vision-KAN | 2024.5.25 | 300(stop) | 73.17 | 93.307 | [Checkpoint](https://drive.google.com/drive/folders/19WPq6bZ9NgX-WxD7qXSTKiHc5D6P8jQP?usp=sharing) |
| CIFAR-100   | 384        | Vision-KAN | 2024.5.25 | 300(stop) | 78.69 | 94.73 | [Checkpoint](https://drive.google.com/drive/folders/1Uhj4yV0HZRQkPFUerxy88B19N1eDdgsc?usp=drive_link) |
| CIFAR-100   | 768        | Vision-KAN | 2024.5.29 | 300(stop) | 79.82 | 95.42 | [Checkpoint](https://drive.google.com/drive/folders/1FT55_6tDO_a135sQKBDn409fDdXvCi4N?usp=drive_link) |

## Latest News 📰

- **5.7.2024**: Released the current Vision KAN code! 🚀 We used efficient KAN to replace the MLP layer in the Transformer block and are pre-training the Tiny model on ImageNet 1k. Updates will be reflected in the table.
- **5.14.2024**: The model is starting to converge! We’re using [192, 20, 192] for input, hidden, and output dimensions.
- **5.15.2024**: Switched from [efficient kan](https://github.com/Blealtan/efficient-kan) to [faster kan](https://github.com/AthanasiosDelis/faster-kan) to double the training speed! 🚀
- **5.16.2024**: Convergence appears to be bottlenecked; considering adjusting the KAN hidden layer size from 20 to 192.
- **5.22.2024**: Fixed Timm version dependency issues and cleaned up the code! 🧹
- **5.24.2024**: Loss decline is slowing, nearing final results! 🔍
- **5.25.2024**: The model with 192 hidden layers is approaching convergence! 🎉 Released the best checkpoint of VisionKAN.

## Architecture 🏗️

We utilized [DeiT](https://github.com/facebookresearch/deit) as the baseline for Vision KAN development. Huge thanks to Meta and MIT for their incredible work! 🙌

## Star History 🌟

[![Star History Chart](https://api.star-history.com/svg?repos=chenziwenhaoshuai/Vision-KAN&type=Date)](https://star-history.com/#chenziwenhaoshuai/Vision-KAN&Date)

## Citation 📑

If you are using our work, please cite:

```bibtex
@misc{VisionKAN2024,
  author = {Ziwen Chen and Gundavarapu and WU DI},
  title = {Vision-KAN: Exploring the Possibility of KAN Replacing MLP in Vision Transformer},
  year = {2024},
  howpublished = {\url{https://github.com/chenziwenhaoshuai/Vision-KAN.git}},
}
```
