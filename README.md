![Latest Release](https://img.shields.io/pypi/v/dflearn.svg)
![License](https://img.shields.io/pypi/l/dflearn.svg)
![Build Status](https://travis-ci.org/founderfan/DFlearn.svg?branch=master)

A data analysis and machine-learning toolset using pandas DataFrame and scikit-learn models.

## Install

```sh
pip install dflearn
```

## Dependencies

- [numPy](http://www.numpy.org): 1.11.0 or higher
- [scipy](https://www.scipy.org/): 0.18.0 or higher
- [pandas](http://pandas.pydata.org/): 0.19.0 or higher
- [statsmodels](http://www.statsmodels.org/): 0.6.0 or higher
- [scikit-learn](http://scikit-learn.org/): 0.18.0 or higher
- [nltk](http://www.nltk.org/): 3.0.0 or higher

## Contents

- MLtools: machine learning tools, main toolset
    - Whole dataset
        - Data summary
            - Variable type, NA/non-NA values, numeric summary statistics, most frequent values.
        - Data cleaning
            - Categorical variables transformation into dummy variables.
            - Numeric variables standarzation/normalization with imputation.
            - Sparse variables deletion.
            - Collinear variables deletion.
    - Machine learning
        - Training/validation set creation
            - Single training/validation set split
            - Cross-validation set creation
            - Cross-join with different models
        - Model training
            - Scikit-learn like regression/classification.
        - Weight analysis
            - Generalized linear model.
            - Tree models variable importance (and random forest interaction).
        - Validation and error analysis
            - Two-way ANOVA and linear regression of CV validation loss.
        - High-dimensional data analysis
            - Heritability of linear mixed model.
            - Bayesian inference of linear model coefficients summary.

- NLtools: natural language tools, waiting for development
    - Clean text
    - Word tokenize

- SNPtools: used for genetic SNP data, not general
    - [PLINK](https://www.cog-genomics.org/plink2) binary data reading
    - Risk score calculation

## License

MIT license