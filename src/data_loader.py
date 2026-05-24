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
        print("File not found.")

    except pd.errors.EmptyDataError:
        print("File is empty.")

    except Exception as e:
        print(f"Unexpected error: {e}")