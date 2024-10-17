# %%
import pandas as pd

out_path = "../data/cleaned/"

# %%
def load_data():
    path = "../data/raw/train.csv"
    return pd.read_csv(path)
df = load_data()

def clean_data(df):
    """
    Clean the housing data by dropping the id column, selecting only the numerical columns, and dropping any rows with missing values.

    Parameters
    ----------
    df : pd.DataFrame
        The housing data to clean.

    Returns
    -------
    pd.DataFrame
        The cleaned housing data.
    """
    df.drop("Id", axis=1, inplace=True)
    df = df.select_dtypes(include=["float64", "int64"])
    df.dropna(inplace=True)
    return df


def save_data(df):
    df.to_csv(out_path + "housing_clean.csv", index=False)

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    save_data(df)

# %%

