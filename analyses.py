import pandas as pd
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

def analyze_signals(df):
    # Moments
    moments = {
        'Media': df.mean(),
        'Varianza': df.var(),
        'Desviación estándar': df.std(),
        'Sesgo': skew(df),
        'Curtosis': kurtosis(df)
    }

    # Linear regression
    for column in df.columns:
        df['Intercept'] = 1.0
        model = sm.OLS(df[column], df[['Intercept'] + [col for col in df.columns if col != column]]).fit()

    # Polynomial regression (2nd degree here)
        df['x2'] = df[column] ** 2
        model2 = sm.OLS(df[column], df[['Intercept', column, 'x2']]).fit()

    fvalue, pvalue = f_oneway(df['Signal1'], df['Signal2'], df['Signal3'], df['Signal4'])

    # LDA
    X = df
    y = ["T1" if "T1" in name else "B1" for name in df.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lda = LDA()
    lda.fit(X_train, y_train)
    predictions = lda.predict(X_test)
    lda_accuracy = accuracy_score(y_test, predictions)

    # Kolmogorov-Smirnov test
    ks_results = {}
    for column in df.columns:
        ks_stat, ks_p = kstest(df[column], 'norm')
        ks_results[column] = ks_p > 0.05  # if True, it's normally distributed

    # Tukey's posthoc test
    tukey_result = pairwise_tukeyhsd(df['Signal1'], ["T1" if "T1" in name else "B1" for name in df.index])

    return {
        "moments": moments,
        "regression": model.summary(),
        "polynomial_regression": model2.summary(),
        "anova_fvalue": fvalue,
        "anova_pvalue": pvalue,
        "LDA_accuracy": lda_accuracy,
        "Kolmogorov_Smirnov_test": ks_results,
        "Tukey_test": str(tukey_result)  # Convert the result to string for easier interpretation
    }
