import pandas as pd


# ==========================
# FEATURE EXPLORATION
# ==========================

def risk_rate(df, col):
    """
    Percentage of high-risk ('yes') observations for each category.
    """
    return (
        df.groupby(col)["CLASS"]
        .value_counts(normalize=True)
        .unstack()["yes"]
        .mul(100)
        .round(2)
    )


def count_rate(df, col):
    """
    Percentage distribution of values including missing values.
    """
    return (
        df[col]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
    )


def count(df, col):
    """
    Count of values including missing values.
    """
    return (
        df[col]
        .value_counts(dropna=False)
    )


# ==========================
# MISSING VALUES
# ==========================

def missing_count(df, col):
    """
    Number of missing values.
    """
    return df[col].isna().sum()


def missing_percent(df, col):
    """
    Percentage of missing values.
    """
    return round(df[col].isna().mean() * 100, 2)


def missing_risk(df, col):
    """
    Compare default rates between missing and non-missing values.
    """
    return (
        df.groupby(df[col].isna())["CLASS"]
        .value_counts(normalize=True)
        .unstack()["yes"]
        .mul(100)
        .round(2)
    )


# ==========================
# CROSSTABS
# ==========================

def cross_count(df, col1, col2):
    """
    Crosstab counts.
    col1 and col2 can be either column names or Series.
    """

    x = df[col1] if isinstance(col1, str) else col1
    y = df[col2] if isinstance(col2, str) else col2

    return pd.crosstab(x, y)


def cross_row_pct(df, col1, col2):
    """
    Row percentages.
    Answers:
    'Within each row, how are observations distributed?'
    """
    return (
        pd.crosstab(
            df[col1],
            df[col2],
            normalize="index"
        )
        .mul(100)
        .round(2)
    )


def cross_col_pct(df, col1, col2):
    """
    Column percentages.
    Answers:
    'Within each column, how are observations distributed?'
    """
    return (
        pd.crosstab(
            df[col1],
            df[col2],
            normalize="columns"
        )
        .mul(100)
        .round(2)
    )


# ==========================
# NUMERICAL FEATURES
# ==========================

def num_summary(df, col):
    """
    Summary statistics for numerical columns.
    """
    return df[col].describe()


def quartile_group(df, col,
                   labels=["Low", "Medium", "High", "Very High"]):
    """
    Create quartile-based bins.
    """
    df[col + "_GROUP"] = pd.qcut(
        df[col],
        q=4,
        labels=labels
    )


def business_bins(df, col, bins, labels):
    """
    Create custom business bins.
    """
    df[col + "_GROUP"] = pd.cut(
        df[col],
        bins=bins,
        labels=labels
    )


# ==========================
# QUICK OVERVIEW
# ==========================

def feature_summary(df, col):
    """
    Quick feature overview.
    """
    print("=" * 50)
    print(f"FEATURE: {col}")
    print("=" * 50)

    print("\nCount")
    print(count(df, col))

    print("\nDistribution (%)")
    print(count_rate(df, col))

    print("\nMissing Count")
    print(missing_count(df, col))

    print("\nMissing Percentage")
    print(missing_percent(df, col))

    print("\nRisk Rate (%)")
    print(risk_rate(df, col))


# ==========================
# TARGET DISTRIBUTION
# ==========================

def target_dist(df):
    """
    Distribution of target variable.
    """
    return (
        df["CLASS"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )


# ==========================
# UNIQUE VALUES
# ==========================

def unique_count(df, col):
    """
    Number of unique values.
    """
    return df[col].nunique()


# ==========================
# DUPLICATES
# ==========================

def duplicate_count(df):
    """
    Number of duplicate rows.
    """
    return df.duplicated().sum()