import pandas as pd


def load_data(file_path):

    try:
        df = pd.read_csv(
            file_path,
            sep="|"
        )

        print("Dataset loaded successfully.")
        return df

    except FileNotFoundError:
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    except pd.errors.EmptyDataError:
        raise ValueError(
            "Dataset file is empty."
        )

    except Exception as e:
        raise Exception(
            f"Unexpected error: {e}"
        )