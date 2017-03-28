A data analysis and machine-learning toolset using pandas DataFrame and scikit-learn models.

## Install

```sh
pip install dflearn
```

## Dependencies

- [numPy](http://www.numpy.org): 1.11.0 or higher
- [scipy](https://www.scipy.org/): 0.18.0 or higher
- [pandas](http://pandas.pydata.org/): 0.19.0 or higher
- [statsmodels](http://www.statsmodels.org/) 0.6.0 or higher
- [sklearn](http://scikit-learn.org/) 0.18.0 or higher
- [nltk](http://www.nltk.org/) 3.0 or higher

## Contents

- MLtools: machine learning tools, main toolset
    - Data cleaning
    - Data summary
    - Training/validation set split
    - Machine learning model training, weight analysis, validation and prediction
    - Cross-validation

- NLtools: natural language tools, waiting for development
    - Clean text
    - Word tokenize

- SNPtools: used for genetic SNP data, not general
    - [PLINK](https://www.cog-genomics.org/plink2) binary data reading
    - Risk score calculation

## License

MIT license