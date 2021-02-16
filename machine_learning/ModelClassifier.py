import sklearn
import numpy
import os
import logging


class ModelClassifier(object):

    def __init__(self):
        self._train_data = None
        self._train_label = None
        self._test_data = None
        self._test_label = None
        self._model = None

    def set_train_test_data(self,
                            train_data,
                            train_label,
                            test_data,
                            test_label):

        self._train_data = train_data
        self._train_label = train_label
        self._test_data = test_data
        self._test_label = test_label

    def evaluate(self, Y, Y_Pred):

        global_balanced_acc = sklearn.metrics.balanced_accuracy_score(
            Y, Y_Pred)

        precision, recall, f1_score, support = sklearn.metrics.precision_recall_fscore_support(
            Y, Y_Pred)

        macro_recall = sum(recall) / 2.0
        macro_precision = sum(precision) / 2.0

        macro_f1_score = (2 * macro_precision * macro_recall) / \
            (macro_precision + macro_recall)

        confusion_matrix = sklearn.metrics.multilabel_confusion_matrix(
            Y, Y_Pred)

        mcc = sklearn.metrics.matthews_corrcoef(Y, Y_Pred)

        iou = sklearn.metrics.jaccard_score(Y, Y_Pred, average=None)

        info = """

        Macro F1 Score : {}
        Balanced accuracy : {}
        Matthews correlation coefficient (MCC) : {}
        IoU : {}

        Noise / Apple :

        PRECISION: {}
        RECALL (Acc 2 class): {}
        F1 Scores: {}
        SUPPORT: {}

        CONFUSION MATRIX:
        {}

        """.format(macro_f1_score,
                   global_balanced_acc,
                   mcc,
                   iou,
                   precision,
                   recall,
                   f1_score,
                   support,
                   confusion_matrix)
        
        print(info, flush=True)
        logging.info(info)

        miou = (iou[0] + iou[1]) / 2

        return macro_f1_score, global_balanced_acc, mcc, f1_score[1], miou

    def evaluate_with_proba(self, Y, Y_pred, proba):

        cond = Y_pred >= proba

        Y_pred_bool = numpy.zeros_like(Y_pred)
        Y_pred_bool[cond] = 1

        return self.evaluate(Y, Y_pred_bool)

    def predict_test(self):
        return self.predict(self._test_data)

    def test(self, proba):

        test_predict = self.predict_test()

        return self.evaluate_with_proba(self._test_label, test_predict, proba)

    def train(self):
        pass

    # def _predict(self, data):
    #     return None

    def predict(self, filename, output_dir, proba=0.50, labeled=False):

        data = numpy.loadtxt(filename)
        if labeled:
            selector = [i for i in range(data.shape[1]) if
                        i not in [0, 1, 2, 6]]
        else:
            selector = [i for i in range(data.shape[1]) if
                        i not in [0, 1, 2]]

        Y = self.predict(data[:, selector])

        cond = Y[:, 1] >= proba
        Y[cond] = 1
        Y[numpy.bitwise_not(cond)] = 0

        res = numpy.column_stack([data[:, :3], Y])
        numpy.savetxt(os.path.join(
            output_dir, os.path.basename(filename)), res)

    def save_predict_proba(self, filename, output_dir, labeled=False):

        data = numpy.loadtxt(filename)
        if labeled:
            selector = [i for i in range(data.shape[1]) if
                        i not in [0, 1, 2, 6]]
        else:
            selector = [i for i in range(data.shape[1]) if
                        i not in [0, 1, 2, 3]]

        proba = numpy.round(self.predict(data[:, selector]),
                            decimals=2)

        res = numpy.column_stack([data[:, :3], proba])
        numpy.savetxt(os.path.join(
            output_dir, os.path.basename(filename)), res)
