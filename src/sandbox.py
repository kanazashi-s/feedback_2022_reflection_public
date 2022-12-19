import pandas as pd


if __name__ == '__main__':
    pseudo_base = pd.read_csv("data/processed/pseudo_1/seed42/pseudo_base.csv")
    vanilla_train = pd.read_csv("data/processed/vanilla/seed42/train.csv")

    print(pseudo_base.shape)
