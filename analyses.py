import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, kstest, f_oneway
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def analyze_labels(df):
    means = df[['Edad', 'Peso (Kg)', 'Estatura (m)']].mean()
    return means.to_dict()

def analyze_signals_fixed_v2(df, label):
    # Moments
    moments = {
        'Media': df.mean().tolist(),
        'Varianza': df.var().tolist(),
        'Desviación estándar': df.std().tolist(),
        'Sesgo': skew(df).tolist(),
        'Curtosis': kurtosis(df).tolist()
    }

    # Define intercept outside the loop
    df['Intercept'] = 1.0

    regression_summaries = {}
    poly_regression_summaries = {}
    for column in df.columns:
        # Exclude 'Intercept' column from looping
        if column != 'Intercept':
            # Linear regression
            independent_vars = ['Intercept'] + [col for col in df.columns if col != column and col != str(column) + '_x2']
            model = sm.OLS(df[column], df[independent_vars]).fit()
            regression_summaries[column] = str(model.summary())

            # Polynomial regression (2nd degree here)
            X_poly = df.copy()
            X_poly[column + '_x2'] = df[column] ** 2
            independent_vars_poly = ['Intercept', str(column) + '_x2'] + [col for col in df.columns if col != column and col != 'Intercept']
            model2 = sm.OLS(df[column], X_poly[independent_vars_poly]).fit()
            poly_regression_summaries[column] = str(model2.summary())

    fvalue, pvalue = f_oneway(df['Signal1'], df['Signal2'], df['Signal3'], df['Signal4'])

    # LDA
    X = df.drop(columns='Intercept')  # Removing the Intercept column
    y = ["T1" if "T1" in label else "B1" for _ in df.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lda = LDA()
    lda.fit(X_train, y_train)
    predictions = lda.predict(X_test)
    lda_accuracy = accuracy_score(y_test, predictions)

    # Kolmogorov-Smirnov test
    ks_results = {}
    for column in df.columns:
        ks_stat, ks_p = kstest(df[column], 'norm')
        ks_results[column] = bool(ks_p > 0.05)

    # Tukey's posthoc test
    if len(set(y)) > 1:
        tukey_result = str(pairwise_tukeyhsd(df['Signal1'], y))
    else:
        tukey_result = "Not applicable (only one group)"

    return {
        "moments": moments,
        "regression": regression_summaries,
        "polynomial_regression": poly_regression_summaries,
        "anova_fvalue": fvalue,
        "anova_pvalue": pvalue,
        "LDA_accuracy": lda_accuracy,
        "Kolmogorov_Smirnov_test": ks_results,
        "Tukey_test": tukey_result
    }

def post_hoc_tukey_analysis(dfs: dict) -> dict:
    values = []
    labels = []

    for filename, df in dfs.items():
        values.extend(df['Signal1'].values.tolist())
        values.extend(df['Signal2'].values.tolist())
        values.extend(df['Signal3'].values.tolist())
        values.extend(df['Signal4'].values.tolist())

        label = filename.split("_")[2]  # Extracts T1, T2, B1 or B2
        labels.extend([label] * len(df) * 4)

    tukey_result = pairwise_tukeyhsd(values, labels)
    result_table = tukey_result.summary().data[1:]

    tukey_dict = {
        "group1": [row[0] for row in result_table],
        "group2": [row[1] for row in result_table],
        "meandiff": [row[2] for row in result_table],
        "p-adj": [row[3] for row in result_table],
        "lower": [row[4] for row in result_table],
        "upper": [row[5] for row in result_table],
        "reject": [int(row[6]) for row in result_table]
    }

    return tukey_dict
