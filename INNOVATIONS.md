# FedSelect experimental innovations

`fedselect_innovative.py` keeps the original FedSelect training path as the
default and exposes two independent, composable switches.

## 1. Client-local Mixup

```powershell
python fedselect_innovative.py --innovation-mixup --mixup-alpha 0.2 ...
```

The Mixup variant blends samples and targets inside each client update. This is
designed to improve generalization from each client's limited local data.

## 2. Multi-view test-time augmentation

```powershell
python fedselect_innovative.py --innovation-multiview-tta --tta-angle 10 ...
```

This averages logits over the original image, its horizontal mirror, and
`+/- tta-angle` rotations of both views. It improves inference stability for
the small CIFAR-10 client models.

## Combined

```powershell
python fedselect_innovative.py `
  --innovation-mixup `
  --innovation-multiview-tta `
  --tta-angle 10 `
  --mixup-alpha 0.2 ...
```

The switches default to false, so omitting both flags preserves the original
mask-update behavior.

The combined two-switch run reaches 82.6563% at communication round 100 on the
reproduction setup, a +0.6152 percentage-point improvement over its matched
reference. See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the full experiment
matrix, including rejected candidates.
