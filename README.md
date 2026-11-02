# [CVPR 2024] Official Repository for "FedSelect: Personalized Federated Learning with Customized Selection of Parameters for Fine-Tuning"

<p>
    <a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/Python-3.7+-1f425f.svg?color=purple">
    </a>
    <a href="https://copyright.illinois.edu/">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
    </a>
</p>

by [Rishub Tamirisa](https://rishub-tamirisa.github.io/research/), [Chulin Xie](https://alphapav.github.io/), [Wenxuan Bao](https://baowenxuan.github.io/), [Andy Zhou](https://www.andyzhou.ai/), [Ron Arel](https://arel.ai/), and [Aviv Shamsian](https://avivsham.github.io/)

See our paper on [arXiv](https://arxiv.org/abs/2404.02478).

## Experimental improvements

This fork adds a separate experimental entry point with two independently
switchable improvements:

1. **Client-local Mixup** during alternating local optimization.
2. **Multi-view test-time augmentation** over the original image, horizontal
   mirror, and +/-10 degree rotations.

```powershell
python fedselect_innovative.py `
  --innovation-mixup `
  --innovation-multiview-tta `
  --mixup-alpha 0.2 `
  --tta-angle 10 `
  --num_users 10 --frac 1.0 --shard_per_user 2 `
  --num_samples 50 --test_size 500 --com_rounds 100 `
  --la_epochs 3 --local_bs 32 --lth_epoch_iters 3 `
  --prune_percent 25 --prune_target 30 --lr 0.05 `
  --clipgradnorm --seed 1
```

The switches default to off, so `fedselect_innovative.py` without either flag
follows the original mask-update behavior.

### CIFAR-10 experiment

All values below use seed 1. The reference and combined experiment run for 100
communication rounds; the multi-view TTA measurement is inference-only on the
same reference checkpoint.

| Method | Round 100 accuracy | Gain vs. reference |
| --- | ---: | ---: |
| FedSelect reference | 82.0410% | - |
| Client-local Mixup | 82.1484% | +0.1074 |
| Multi-view TTA on reference checkpoint | 82.4512% | +0.4102 |
| **Mixup + multi-view TTA** | **82.6563%** | **+0.6152** |

Gradient clipping is enabled in these experiments to prevent the unclipped
run from producing NaN parameters under this PyTorch/CUDA stack. See
[`INNOVATIONS.md`](INNOVATIONS.md) for switch descriptions.

## Citation

```
@misc{tamirisa2024fedselectpersonalizedfederatedlearning,
      title={FedSelect: Personalized Federated Learning with Customized Selection of Parameters for Fine-Tuning}, 
      author={Rishub Tamirisa and Chulin Xie and Wenxuan Bao and Andy Zhou and Ron Arel and Aviv Shamsian},
      year={2024},
      eprint={2404.02478},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2404.02478}, 
}
```
