import numpy as np
import math

class SegmentationMetrics:
    def __init__(self, ground_truth, predicted):
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

    def _ratio(self, num, den, both_empty=0.0):
        """num / den. When den is 0, returns `both_empty` if both masks are empty, else 0.0."""
        if den == 0:
            return both_empty if self.both_empty else 0.0
        return float(num) / float(den)

    def recall(self):
        """
        Recall, aka true positive rate or sensitivity: TP / (TP + FN).
        Returns 1.0 when both masks are empty and 0.0 when only the ground truth is empty.
        """
        return self._ratio(self.tp, self.tp + self.fn, both_empty=1.0)

    def accuracy(self):
        """
        Accuracy: (TP + TN) / number of voxels. Not used in the paper.
        Returns 0.0 for empty arrays.
        """
        return self._ratio(self.tp + self.tn, self.positives + self.negatives)

    def specificity(self):
        """
        Specificity, aka true negative rate: TN / (TN + FP). Not used in the paper.
        Returns 0.0 when the ground truth has no background voxels.
        """
        return self._ratio(self.tn, self.tn + self.fp)

    def fall_out(self):
        """
        Fall-out, aka false positive rate: FP / (FP + TN). Not used in the paper.
        Returns 0.0 when the ground truth has no background voxels.
        """
        return self._ratio(self.fp, self.fp + self.tn)

    def precision(self):
        """
        Precision: TP / (TP + FP).
        Returns 1.0 when both masks are empty and 0.0 when only the prediction is empty.
        """
        return self._ratio(self.tp, self.tp + self.fp, both_empty=1.0)

    def f_score(self):
        """
        F-score: 2pr / (p + r), equal to the Dice coefficient for binary masks.
        Returns 1.0 when both masks are empty.
        """
        return self.dice()

    def jaccard(self):
        """
        Jaccard index: TP / (TP + FP + FN), the intersection over the union.
        Returns 1.0 when both masks are empty.
        """
        return self._ratio(self.tp, self.tp + self.fp + self.fn, both_empty=1.0)

    def dice(self):
        """
        Dice coefficient, aka F-score: 2TP / (2TP + FP + FN).
        Returns 1.0 when both masks are empty.
        """
        return self._ratio(2*self.tp, 2*self.tp + self.fp + self.fn, both_empty=1.0)
