import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

class ReportBuilder:
    def classification_report(self, y_true, y_pred):
        return classification_report(y_true, y_pred, output_dict=True)

    def confusion(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)
