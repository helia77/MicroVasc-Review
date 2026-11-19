import numpy as np
import math

class SegmentationMetrics:
    """
    Voxel-wise comparison of a binary prediction with a binary ground truth.

    The counts are computed once, when the object is created. Any nonzero value counts as
    foreground.

    Args:
        ground_truth: Ground truth mask.
        predicted: Predicted mask, same shape as the ground truth.

    Attributes:
        vol_true: The ground truth mask.
        vol_pred: The predicted mask.
        positives: Number of foreground voxels in the ground truth.
        negatives: Number of background voxels in the ground truth.
        tp: True positives.
        tn: True negatives.
        fp: False positives.
        fn: False negatives.
        both_empty: True if neither mask has a foreground voxel.

    Raises:
        ValueError: If the two masks have different shapes.

    Examples:
        >>> import numpy as np
        >>> from microvasc.metrics.geometry import SegmentationMetrics
        >>> ground_truth = np.array([1, 1, 1, 0, 0, 0])
        >>> predicted = np.array([1, 1, 0, 1, 0, 0])
        >>> m = SegmentationMetrics(ground_truth, predicted)
        >>> m.jaccard(), m.dice()
        (0.5, 0.6666666666666666)
    """

    def __init__(self, ground_truth: np.ndarray, predicted: np.ndarray) -> None:
        if np.shape(ground_truth) != np.shape(predicted):
            raise ValueError(f"ground_truth and predicted must have the same shape, "
                             f"got {np.shape(ground_truth)} and {np.shape(predicted)}")
        self.vol_true = ground_truth
        self.vol_pred = predicted
        self.positives = np.count_nonzero(ground_truth)
        self.negatives = np.count_nonzero(np.logical_not(ground_truth))

        self.tp = np.logical_and(ground_truth, predicted).sum()
        self.tn = np.logical_and(np.logical_not(predicted), np.logical_not(ground_truth)).sum()
        self.fp = np.logical_and(np.logical_not(ground_truth), predicted).sum()
        self.fn = np.logical_and(ground_truth, np.logical_not(predicted)).sum()
        # no foreground in either mask
        self.both_empty = (self.tp + self.fp + self.fn) == 0

    def _ratio(self, num, den, both_empty: float = 0.0) -> float:
        """num / den. When den is 0, returns `both_empty` if both masks are empty, else 0.0."""
        if den == 0:
            return both_empty if self.both_empty else 0.0
        return float(num) / float(den)

    def recall(self) -> float:
        """
        Recall, also called true positive rate or sensitivity: TP / (TP + FN).

        Returns:
            Recall in [0, 1]. 1.0 when both masks are empty, 0.0 when only the ground truth
                is empty.
        """
        return self._ratio(self.tp, self.tp + self.fn, both_empty=1.0)

    def accuracy(self) -> float:
        """
        Accuracy: (TP + TN) / number of voxels. Not used in the paper.

        Returns:
            Accuracy in [0, 1]. 0.0 for arrays with no voxels.
        """
        return self._ratio(self.tp + self.tn, self.positives + self.negatives)

    def specificity(self) -> float:
        """
        Specificity, also called true negative rate: TN / (TN + FP). Not used in the paper.

        Returns:
            Specificity in [0, 1]. 0.0 when the ground truth has no background voxels.
        """
        return self._ratio(self.tn, self.tn + self.fp)

    def fall_out(self) -> float:
        """
        Fall-out, also called false positive rate: FP / (FP + TN). Not used in the paper.

        Returns:
            Fall-out in [0, 1]. 0.0 when the ground truth has no background voxels.
        """
        return self._ratio(self.fp, self.fp + self.tn)

    def precision(self) -> float:
        """
        Precision: TP / (TP + FP).

        Returns:
            Precision in [0, 1]. 1.0 when both masks are empty, 0.0 when only the prediction
                is empty.
        """
        return self._ratio(self.tp, self.tp + self.fp, both_empty=1.0)

    def f_score(self) -> float:
        """
        F-score: 2PR / (P + R), equal to the Dice coefficient for binary masks.

        Returns:
            F-score in [0, 1]. 1.0 when both masks are empty.
        """
        return self.dice()

    def jaccard(self) -> float:
        """
        Jaccard index: TP / (TP + FP + FN), the intersection over the union.

        Returns:
            Jaccard index in [0, 1]. 1.0 when both masks are empty.
        """
        return self._ratio(self.tp, self.tp + self.fp + self.fn, both_empty=1.0)

    def dice(self) -> float:
        """
        Dice coefficient: 2TP / (2TP + FP + FN), equal to the F-score.

        Returns:
            Dice coefficient in [0, 1]. 1.0 when both masks are empty.
        """
        return self._ratio(2*self.tp, 2*self.tp + self.fp + self.fn, both_empty=1.0)
