import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.

    :param int patience: Number of epochs with no improvement after which training stops.
    :param float min_delta: Minimum change to qualify as an improvement (default: 0.0).
    :param bool maximize: Whether to maximize the metric (True) or minimize it (False).
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, maximize: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.maximize = maximize
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.

        :param float score: Current validation metric score.
        :param int epoch: Current epoch number.
        :return: True if training should stop, False otherwise.
        :rtype: bool
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.maximize:
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(
                    f"Early stopping triggered. Best score: {self.best_score:.4f} "
                    f"at epoch {self.best_epoch}"
                )
                return True

        return False
