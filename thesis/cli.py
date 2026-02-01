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
        "--skip-ica",
        action="store_true",
        help="Skip ICA artifact removal (faster but less clean data)",
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
        + ["all"],
        default="all",
        help="Channel to use: specific channel name (Fp1, T7, etc.) or 'all' for all 8 channels. "
        "When 'all' is selected, model receives 8-channel spectrograms. "
        "Single channel selection automatically skips ICA.",
    )
    preproc_group.add_argument(
        "--class-mode",
        type=str,
        default="2",
        choices=["2", "4"],
        help="Classification mode: "
        "'2' forces binary (healthy vs any-pathological), "
        "'4' requires --dataset both for all classes. "
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
        help="Model architecture to use: 'CNN_LSTM_DepCap' (default, full model) "
        "or 'Smaller' (reduced model with ~50%% fewer parameters)",
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
        choices=["mdd", "cane", "both"],
        help="Dataset to train on: 'mdd' (2 classes: normal, mdd), "
        "'cane' (2-3 classes: normal, anxious[, mdd]), "
        "or 'both' (3 classes: normal, mdd, anxious)",
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

    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Run inference on a single EDF file")
    add_preprocessing_args(run_parser)
    run_parser.add_argument("model_path", type=str, help="Path to trained model checkpoint (.pth)")
    run_parser.add_argument("edf_file", type=str, help="Path to EDF file to classify")
    add_model_args(run_parser)
    run_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on",
    )

    return parser
