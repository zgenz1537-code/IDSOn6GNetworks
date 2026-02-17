if True:
    from reset_random import reset_random

    reset_random()
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import CLASSES, print_df_to_table

pd.set_option("use_inf_as_na", True)


def show_count(df, class_col):
    print("[INFO] Class Distribution")
    cvc = df[class_col].value_counts(sort=False)[CLASSES]
    sdf = cvc.to_frame()
    sdf.insert(0, "cls", cvc.index)
    sdf.columns = ["Class", "Count"]
    print_df_to_table(sdf)
    return df


def replace_categorical_cols(df):
    for col in df.columns:
        if col == "class":
            continue
        if df[col].dtype != "object":
            continue
        print("[INFO] Replacing Categorical Values in Column :: {0}".format(col))
        repd = {v: k + 1 for k, v in enumerate(sorted(df[col].unique()))}
        df[col].replace(repd, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_dataset():
    print("[INFO] Working on ToN-IoT Dataset")
    df = pd.read_csv("Data/TON_IoT_Train_Test_Network.csv")
    df.dropna(inplace=True)
    df.drop(["label"], axis=1, inplace=True)
    df.columns = df.columns.tolist()[:-1] + ["class"]
    print(df["class"].value_counts())
    df["class"].replace(
        {
            "normal": "Normal",
            "scanning": "Scanning",
            "mitm": "MiTM",
            "dos": "DoS",
            "ddos": "DDoS",
            "password": "Password",
            "injection": "Injection",
            "xss": "XSS",
            "ransomware": "Ransomware",
            "backdoor": "Backdoor",
        },
        inplace=True,
    )
    print(df["class"].value_counts(sort=False))
    dfs = []
    for c in CLASSES:
        sdf = df[df["class"] == c].head(20000)
        dfs.append(sdf)
    df = pd.concat(dfs)
    for c in df.columns:
        try:
            df[c] = df[c].astype(float)
        except:
            pass
    df = replace_categorical_cols(df)
    x, y = df.values[:, :-1], df.values[:, -1]
    print("[INFO] Normalizing Data")
    mm = StandardScaler()
    x = mm.fit_transform(x, y)
    df = pd.DataFrame(np.concatenate([x, y.reshape(-1, 1)], axis=1), columns=df.columns)
    show_count(df, "class")
    df["class"].replace({c: i for i, c in enumerate(CLASSES)}, inplace=True)
    dp = "Data/preprocessed.csv"
    print("[INFO] Data Shape After Preprocessed :: {0}".format(df.shape))
    print("[INFO] Saving Preprocessed Data :: {0}".format(dp))
    df.to_csv(dp, index=False)


if __name__ == "__main__":
    load_dataset()
