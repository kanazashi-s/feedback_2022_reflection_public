from pathlib import Path
from config.general import GeneralCFG
from data import load_data
from preprocess.cv import add_fold_column


def make():
    output_path = Path("/workspace", "data", "processed", "paragraph")
    output_path.mkdir(parents=True, exist_ok=True)

    # load all csv files
    train = load_data.train()
    test = load_data.test()
    sample_submission = load_data.sample_submission()

    # [Important] replace \n\n to | in full_text to explicitly separate paragraphs
    train["full_text"] = train["full_text"].str.replace("\n\n", " | ")
    test["full_text"] = test["full_text"].str.replace("\n\n", " | ")

    # create sample_oof
    sample_oof = train.drop(columns=["full_text"])
    sample_oof.iloc[:, 1:] = 3.0

    # prepare 3 patterns of cv
    for seed in GeneralCFG.seeds:
        fold_output_path = output_path / f"seed{seed}"
        fold_output_path.mkdir(exist_ok=True)

        train_fold = add_fold_column(train, num_folds=GeneralCFG.n_fold, random_state=seed)

        # save
        train_fold.to_csv(fold_output_path / "train.csv", index=False)
        test.to_csv(fold_output_path / "test.csv", index=False)
        sample_submission.to_csv(fold_output_path / "sample_submission.csv", index=False)
        sample_oof.to_csv(fold_output_path / "sample_oof.csv", index=False)


if __name__ == "__main__":
    make()
