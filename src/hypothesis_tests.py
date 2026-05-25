from scipy.stats import ttest_ind


def run_ttest(group_a, group_b):

    statistic, p_value = ttest_ind(
        group_a,
        group_b,
        equal_var=False,
        nan_policy="omit"
    )

    return statistic, p_value