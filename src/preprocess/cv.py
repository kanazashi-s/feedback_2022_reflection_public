import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from config.general import GeneralCFG


def add_fold_column(input_df, num_folds=5, random_state=42):
    output_df = input_df.copy()
    output_df['fold'] = -1

    # https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/368437
    y = pd.get_dummies(data=output_df[GeneralCFG.target_cols], columns=GeneralCFG.target_cols)
    kf = MultilabelStratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=output_df, y=y)):
        output_df.loc[val_idx, 'fold'] = fold

    return output_df
