import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss_ratio_by_province(df):

    province_loss = (
        df.groupby("Province")["LossRatio"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10,5))

    sns.barplot(
        x=province_loss.index,
        y=province_loss.values
    )

    plt.title("Average Loss Ratio by Province")

    plt.xticks(rotation=45)

    plt.show()


def plot_gender_loss_ratio(df):

    gender_loss = (
        df.groupby("Gender")["LossRatio"]
        .mean()
    )

    plt.figure(figsize=(6,4))

    sns.barplot(
        x=gender_loss.index,
        y=gender_loss.values
    )

    plt.title("Average Loss Ratio by Gender")

    plt.show()