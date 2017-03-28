A data analysis and machine-learning toolset using pandas DataFrame and scikit-learn models.

|Latest Release| |Package Status| |License| |PyPI|

Install
=======

.. code:: sh

    pip install dflearn

Dependencies
============

-  `numPy <http://www.numpy.org>`__: 1.11.0 or higher
-  `scipy <https://www.scipy.org/>`__: 0.18.0 or higher
-  `pandas <http://pandas.pydata.org/>`__: 0.19.0 or higher
-  `statsmodels <http://www.statsmodels.org/>`__: 0.6.0 or higher
-  `sklearn <http://scikit-learn.org/>`__: 0.18.0 or higher
-  `nltk <http://www.nltk.org/>`__: 3.0 or higher

Contents
========

-  MLtools: machine learning tools, main toolset

   -  Data cleaning
   -  Data summary
   -  Training/validation set split
   -  Machine learning model training, weight analysis, validation and prediction
   -  Cross-validation

-  NLtools: natural language tools, waiting for development

   -  Clean text
   -  Word tokenize

-  SNPtools: used for genetic SNP data, not general

   -  `PLINK <https://www.cog-genomics.org/plink2>`__ binary data reading
   -  Risk score calculation

License
=======

MIT license

.. |Latest Release| image:: https://img.shields.io/pypi/v/dflearn.svg
.. |Package Status| image:: https://img.shields.io/pypi/status/dflearn.svg
.. |License| image:: https://img.shields.io/pypi/l/dflearn.svg
.. |PyPI| image:: https://img.shields.io/pypi/dm/dflearn.svg
