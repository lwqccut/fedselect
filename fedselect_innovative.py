"""Experimental FedSelect entry point with independently switchable innovations.

Use ``--innovation-mixup`` and/or ``--innovation-multiview-tta``. Both
switches default to off, so this script also provides an unchanged baseline.
"""

from fedselect import load_model, lth_args_parser, run_base_experiment, setup_seed


def main() -> None:
    args = lth_args_parser()
    setup_seed(args.seed)
    model = load_model(args)
    print(
        "Innovations:",
        f"mixup={args.innovation_mixup}",
        f"mixup_alpha={args.mixup_alpha}",
        f"multiview_tta={args.innovation_multiview_tta}",
        f"tta_angle={args.tta_angle}",
        flush=True,
    )
    run_base_experiment(model, args)


if __name__ == "__main__":
    main()
