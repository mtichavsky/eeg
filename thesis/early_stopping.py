import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.

    Tracks actual epochs (not evaluation calls) since last improvement.
    When patience is reached, training should stop.

    :param int patience: Number of epochs with no improvement after which training stops.
    :param float min_delta: Minimum change to qualify as an improvement (default: 0.0).
    :param bool maximize: Whether to maximize the metric (True) or minimize it (False).
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, maximize: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.maximize = maximize
        self.best_score: float | None = None
        self.early_stop = False
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def __call__(self, score: float, current_epoch: int) -> bool:
        """
        Check if training should stop based on validation score.

        This should be called only when validation is performed. The method tracks
        the actual number of epochs since the last improvement, not the number of
        validation calls.

        :param float score: Current validation metric score.
        :param int current_epoch: Current epoch number (1-indexed).
        :return: True if training should stop, False otherwise.
        :rtype: bool
        """
        if self.best_score is None:
            # First evaluation
            self.best_score = score
            self.best_epoch = current_epoch
            self.epochs_without_improvement = 0
            logger.info(
                f"Early stopping initialized: best score {self.best_score:.4f} "
                f"at epoch {self.best_epoch}"
            )
            return False

        # Calculate epochs since last improvement
        self.epochs_without_improvement = current_epoch - self.best_epoch

        if self.maximize:
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            logger.info(
                f"Validation improved from {self.best_score:.4f} to {score:.4f} "
                f"(+{score - self.best_score:.4f}) at epoch {current_epoch}"
            )
            self.best_score = score
            self.best_epoch = current_epoch
            self.epochs_without_improvement = 0
        else:
            logger.debug(
                f"No improvement for {self.epochs_without_improvement} epochs "
                f"(best: {self.best_score:.4f} at epoch {self.best_epoch}, "
                f"current: {score:.4f})"
            )
            if self.epochs_without_improvement >= self.patience:
                self.early_stop = True
                logger.info(
                    f"Early stopping triggered after {self.epochs_without_improvement} epochs "
                    f"without improvement. Best score: {self.best_score:.4f} "
                    f"at epoch {self.best_epoch}"
                )
                return True

        return False
