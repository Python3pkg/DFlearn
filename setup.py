from distutils.core import setup
setup(
    name = 'dflearn',
    packages = ['dflearn'], # this must be the same as the name above
    version = '0.1.2',
    install_requires=[
        'numpy>=1.11.0',
        'scipy>=0.18.0',
        'pandas>=0.19.0',
        'statsmodels>=0.6.0',
        'scikit-learn>=0.18.0',
        'nltk>=3.0.0'
    ],
    test_suite='tests',
    description = 'A DataFrame-based Machine Learning Toolset in Python',
    author = 'Fangda Fan',
    author_email = 'recreating@outlook.com',
    url = 'https://github.com/founderfan/DFlearn', # use the URL to the github repo
    download_url = 'https://github.com/founderfan/DFlearn/archive/0.1.2.tar.gz', # I'll explain this in a second
    keywords = 'machine-learning pandas scikit-learn cross-validation', # arbitrary keywords
    classifiers = ['Programming Language :: Python :: 3'],
    license='MIT'
)