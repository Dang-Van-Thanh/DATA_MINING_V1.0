import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

class AssociationMiner:
    def __init__(self, min_support, min_confidence, min_lift):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift

    def run(self, df_onehot: pd.DataFrame) -> pd.DataFrame:
        freq = apriori(
            df_onehot,
            min_support=self.min_support,
            use_colnames=True
        )

        rules = association_rules(
            freq,
            metric="confidence",
            min_threshold=self.min_confidence
        )

        rules = rules[rules["lift"] >= self.min_lift]
        rules = rules.sort_values("lift", ascending=False)

        return rules
