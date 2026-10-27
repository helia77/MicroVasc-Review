import numpy as np
import math

class metric:
    def __init__(self, true, predicted):
        self.vol_true = true
        self.vol_pred = predicted
        self.positives = np.count_nonzero(true)
        self.negatives = np.count_nonzero(np.logical_not(true))

        self.TP = np.logical_and(true, predicted).sum()
        self.TN = np.logical_and(np.logical_not(predicted), np.logical_not(true)).sum()
        self.FP = np.logical_and(np.logical_not(true), predicted).sum()
        self.FN = np.logical_and(true, np.logical_not(predicted)).sum()
        # no foreground in either mask
        self.both_empty = (self.TP + self.FP + self.FN) == 0

    def _ratio(self, num, den, both_empty=0.0):
        """num / den. When den is 0, returns `both_empty` if both masks are empty, else 0.0."""
        if den == 0:
            return both_empty if self.both_empty else 0.0
        return float(num) / float(den)

    def TPR(self):
        """
        True positive rate, aka recall or sensitivity: TP / (TP + FN).
        Returns 1.0 when both masks are empty and 0.0 when only the ground truth is empty.
        """
        return self._ratio(self.TP, self.TP + self.FN, both_empty=1.0)

    def recall(self):
        """
        Recall: TP / (TP + FN), same as TPR().
        Returns 1.0 when both masks are empty and 0.0 when only the ground truth is empty.
        """
        return self.TPR()

    def FPR(self):
        """
        False positive rate: FP / (FP + TN). Not used in the paper.
        Returns 0.0 when the ground truth has no background voxels.
        """
        return self._ratio(self.FP, self.negatives)

    def accuracy(self):
        """
        Accuracy: (TP + TN) / number of voxels. Not used in the paper.
        Returns 0.0 for empty arrays.
        """
        return self._ratio(self.TP + self.TN, self.positives + self.negatives)

    def specificity(self):
        """
        Specificity, aka true negative rate: TN / (TN + FP). Not used in the paper.
        Returns 0.0 when the ground truth has no background voxels.
        """
        return self._ratio(self.TN, self.TN + self.FP)

    def fall_out(self):
        """
        Fall-out, aka false positive rate: FP / (FP + TN). Not used in the paper.
        Returns 0.0 when the ground truth has no background voxels.
        """
        return self._ratio(self.FP, self.FP + self.TN)

    def precision(self):
        """
        Precision: TP / (TP + FP).
        Returns 1.0 when both masks are empty and 0.0 when only the prediction is empty.
        """
        return self._ratio(self.TP, self.TP + self.FP, both_empty=1.0)

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
        return self._ratio(self.TP, self.TP + self.FP + self.FN, both_empty=1.0)

    def dice(self):
        """
        Dice coefficient, aka F-score: 2TP / (2TP + FP + FN).
        Returns 1.0 when both masks are empty.
        """
        return self._ratio(2*self.TP, 2*self.TP + self.FP + self.FN, both_empty=1.0)
