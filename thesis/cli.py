"""Command-line interface argument parsing for EEG classification training and inference."""

import argparse

from thesis.model import MODEL_REGISTRY


def add_preprocessing_args(parser: argparse.ArgumentParser) -> None:
    """
    Add shared preprocessing arguments to a parser.

    :param argparse.ArgumentParser parser: Parser or subparser to add arguments to.
    """
    # Import here to avoid circular dependencies
    from thesis.dataset import CANEDataset, MDDDataset

    preproc_group = parser.add_argument_group("Preprocessing options")
    preproc_group.add_argument(
        "--condition",
        type=str,
        default="EC",
        help="EEG condition to use (case-insensitive). Options: EC, EO, TASK (MDD only), "
        "or EC+EO to train on both eyes closed and eyes open together. "
        "When using EC+EO, subject EC and EO recordings stay in the same fold but are "
        "evaluated as separate recordings. "
        "Examples: --condition ec, --condition ec+eo",
    )
    preproc_group.add_argument(
        "--skip-artifact-removal",
        action="store_true",
        help="Skip artifact interpolation and clipping in CANE dataset "
        "(let the neural network learn to handle artifacts)",
    )
    preproc_group.add_argument(
        "--channel",
        type=str,
        choices=list(
            set(MDDDataset.CHANNEL_MAPPING.values()) | set(CANEDataset.CHANNEL_MAPPING.values())
        )
        + ["all", "in-ear"],
        default="all",
        help="Channel to use: specific channel name (Fp1, T7, etc.), 'all' for all 8 channels, "
        "or 'in-ear' for in-ear EEG. When 'in-ear' is selected, CANE is replaced with real "
        "IDUN in-ear recordings; MDD and SAD use synthetic bipolar derivation T8-T7. "
        "Includes 50%% sign flip augmentation. "
        "When 'all' is selected, model receives 8-channel spectrograms.",
    )
    preproc_group.add_argument(
        "--class-mode",
        type=str,
        default="2",
        choices=["2", "4"],
        help="Classification mode: "
        "'2' forces binary (healthy vs any-pathological), "
        "'4' requires --dataset all for all classes. "
        "Note: not all datasets contain all classes.",
    )
    preproc_group.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode: load only one file from each class (hardcoded filenames) "
        "for faster debugging. Skips full dataset preprocessing.",
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """
    Add shared model selection argument to a parser.

    :param argparse.ArgumentParser parser: Parser or subparser to add arguments to.
    """
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="CNN_LSTM_DepCap",
        choices=MODEL_REGISTRY.keys(),
        help="Model architecture to use: 'CNN_LSTM_DepCap' (default, full model), "
        "'Smaller' (reduced LSTM model), 'SmallerAll' (multi-channel LSTM), "
        "'SmallerAttn' (reduced model with self-attention), "
        "or 'SmallerAllAttn' (multi-channel with self-attention)",
    )


def get_arg_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for the EEG classification CLI.

    :return: Configured argument parser with train and run subcommands.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="EEG Classification Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the model using cross-validation")
    add_preprocessing_args(train_parser)
    train_parser.add_argument(
        "--dataset",
        type=str,
        default="mdd",
        choices=["mdd", "cane", "sad", "all"],
        help="Dataset to train on: 'mdd' (2 classes: normal, mdd), "
        "'cane' (2-4 classes: normal, anxious, mdd, comorbid), "
        "'sad' (2 classes: normal, anxious; same channel setup as MDD), "
        "or 'all' (combined datasets with 2-4 classes)",
    )
    add_model_args(train_parser)
    train_parser.add_argument(
        "--n-folds", type=int, default=10, help="Number of cross-validation folds"
    )
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    train_parser.add_argument(
        "--epochs", type=int, default=100, help="Maximum number of epochs per fold"
    )
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--dropout", type=float, default=0.5, help="Dropout")
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization)",
    )
    train_parser.add_argument(
        "--l1-lambda",
        type=float,
        default=0.0,
        help="L1 regularization coefficient added to the training loss as "
        "lambda × sum(|params|). Default: 0.0 (disabled). "
        "TSception paper uses 1e-6.",
    )
    train_parser.add_argument(
        "--no-cosine-lr",
        dest="cosine_lr",
        action="store_false",
        help="Disable cosine annealing LR schedule (use constant LR instead). "
        "Cosine annealing is enabled by default for all models.",
    )
    train_parser.add_argument(
        "--focal-loss",
        action="store_true",
        help="Use Focal Loss instead of cross-entropy. Class weights computed per fold "
        "are passed as alpha. Compatible with WeightedRandomSampler.",
    )
    train_parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focusing exponent for Focal Loss (only used with --focal-loss). "
        "gamma=0 reduces to weighted cross-entropy. Typical range: 1.0-5.0.",
    )
    train_parser.add_argument("--val-every", type=int, default=2, help="Validate every N epochs")
    train_parser.add_argument(
        "--save-every", type=int, default=100, help="Save checkpoint every N epochs"
    )
    train_parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    train_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )

    train_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to train on",
    )

    # Data augmentation arguments
    train_parser.add_argument(
        "--augment-data",
        type=float,
        default=None,
        metavar="PROB",
        help="Enable data augmentation with given probability (0.0-1.0). "
        "Example: --augment-data 0.5 applies each augmentation with 50%% chance. "
        "Augmentations: FTSurrogate, MagWarp, TimeReverse, Scaling.",
    )

    # Transfer learning arguments
    train_parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint (.pth) for transfer learning. "
        "Must use same --model architecture as pretrained model.",
    )
    train_parser.add_argument(
        "--freeze-cnn",
        action="store_true",
        help="Freeze CNN layers (conv1, conv2) during training. "
        "Can be used alone or with --freeze-lstm.",
    )
    train_parser.add_argument(
        "--freeze-lstm",
        action="store_true",
        help="Freeze LSTM layer during training. Can be used alone or with --freeze-cnn.",
    )

    # Deformer hyperparameters (ignored for non-Deformer models)
    deformer_group = train_parser.add_argument_group(
        "Deformer hyperparameters",
        description="Only used when --model Deformer or --model DeformerS is selected. "
        "Defaults are model-specific: Deformer uses paper defaults (depth=4, heads=16, "
        "num_kernel=64); DeformerS uses reduced defaults (depth=3, heads=4, num_kernel=48). "
        "Passing any flag overrides the model's default for that parameter only.",
    )
    deformer_group.add_argument(
        "--chunk-duration",
        type=float,
        default=10.0,
        metavar="SECS",
        help="EEG chunk duration in seconds (default: 10.0). "
        "Applies only to raw EEG models (Deformer/DeformerS); "
        "spectrogram models always use 10 s. "
        "Use 5.0 to halve model size and double training chunks.",
    )
    deformer_group.add_argument(
        "--deformer-depth",
        type=int,
        default=None,
        help="Number of transformer layers. Default: model-specific "
        "(Deformer=4, DeformerS=3).",
    )
    deformer_group.add_argument(
        "--deformer-heads",
        type=int,
        default=None,
        help="Number of attention heads. Default: model-specific "
        "(Deformer=16, DeformerS=4).",
    )
    deformer_group.add_argument(
        "--deformer-num-kernel",
        type=int,
        default=None,
        help="Number of CNN kernels in shallow encoder. Default: model-specific "
        "(Deformer=64, DeformerS=48).",
    )
    deformer_group.add_argument(
        "--deformer-mlp-dim",
        type=int,
        default=None,
        help="FeedForward hidden dimension. Default: 16 (same for all variants).",
    )
    deformer_group.add_argument(
        "--deformer-dim-head",
        type=int,
        default=None,
        help="Dimension per attention head. Default: 16 (same for all variants).",
    )
    deformer_group.add_argument(
        "--deformer-temporal-kernel",
        type=int,
        default=None,
        help="Temporal kernel size for CNN encoder (must be odd). "
        "Paper formula: Odd(0.1 × sampling_rate). Default: 25 for 250 Hz.",
    )

    # LGGNet hyperparameters (ignored for non-LGGNet models)
    lggnet_group = train_parser.add_argument_group(
        "LGGNet hyperparameters",
        description="Only used when --model LGGNet or --model LGGNetS is selected. "
        "Defaults are model-specific (LGGNet: num_T=64, LGGNetS: num_T=32; all other defaults "
        "are shared). Passing any flag overrides the model's default for that parameter only.",
    )
    lggnet_group.add_argument(
        "--lggnet-num-t",
        type=int,
        default=None,
        help="Temporal filters per branch. Default: 64 (LGGNet) or 32 (LGGNetS).",
    )
    lggnet_group.add_argument(
        "--lggnet-out-graph",
        type=int,
        default=None,
        help="GCN output features. Default: 32.",
    )
    lggnet_group.add_argument(
        "--lggnet-pool",
        type=int,
        default=None,
        help="PowerLayer pooling window. Paper uses 16 at 128 Hz; default here is 32 at 250 Hz.",
    )
    lggnet_group.add_argument(
        "--lggnet-pool-step-rate",
        type=float,
        default=None,
        help="Pool stride as a fraction of pool size (stride = pool × rate). Default: 0.25.",
    )
    lggnet_group.add_argument(
        "--lggnet-graph-type",
        type=str,
        default=None,
        choices=["frontal", "hemisphere"],
        help="Graph topology. 'hemisphere': L{Fp1,T7,C3}/R{Fp2,T8,C4}/Mid{Cz,Oz} (default). "
        "'frontal': {Fp1,Fp2}/{T7,T8}/{C3,C4,Cz}/{Oz}.",
    )

    # TSception hyperparameters (ignored for non-TSception models)
    tsception_group = train_parser.add_argument_group(
        "TSception hyperparameters",
        description="Only used when --model TSception or --model TSceptionS is selected. "
        "TSception uses hidden=128 (paper default, ~1.05 M params); "
        "TSceptionS uses hidden=32 (paper's cross-dataset recommendation, ~264 K params). "
        "Passing any flag overrides the model's default for that parameter only.",
    )
    tsception_group.add_argument(
        "--tsception-num-t",
        type=int,
        default=None,
        help="Temporal inception filter count. Default: 9 (both variants, paper value).",
    )
    tsception_group.add_argument(
        "--tsception-num-s",
        type=int,
        default=None,
        help="Spatial filter count per branch. Default: 6 (both variants, paper value).",
    )
    tsception_group.add_argument(
        "--tsception-hidden",
        type=int,
        default=None,
        help="FC hidden layer nodes. Default: 128 (TSception) / 32 (TSceptionS).",
    )
    tsception_group.add_argument(
        "--tsception-sampling-rate",
        type=int,
        default=None,
        help="Sampling rate in Hz used to compute temporal kernel sizes as "
        "int(fraction × rate) for fractions [0.5, 0.25, 0.125]. Default: 250 Hz.",
    )

    # Run subcommand
    run_parser = subparsers.add_parser(
        "run", help="Run inference on a single EEG file (.edf or .csv)"
    )
    add_preprocessing_args(run_parser)
    run_parser.add_argument("model_path", type=str, help="Path to trained model checkpoint (.pth)")
    run_parser.add_argument("file", type=str, help="Path to EEG file to classify (.edf or .csv)")
    add_model_args(run_parser)
    run_parser.add_argument(
        "--fs",
        type=int,
        default=None,
        dest="fs",
        help="Sampling rate in Hz (auto-detected from file if omitted)",
    )
    run_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on",
    )

    return parser
