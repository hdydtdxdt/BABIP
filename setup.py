from setuptools import setup, find_packages

setup(
    name='BABIP',
    version='0.0.1',
    description='Easy Tools for Improving Machine Learning Models',
    author='hdydtdxdt',
    author_email='hdydtdxdt@gmail.com',
    url='https://github.com/hdydtdxdt/BABIP',
    install_requires=['scikit-learn',],
    packages=find_packages(exclude=[]),
    keywords=['machine learning', 'deep learning', 'artificial intelligence', 'korean', 'easy', 'hdydtdxdt'],
    python_requires='>=3.6',
    package_data={
        'babip' : [
            'koda/wordnet.pickle',
            'rec/stopwords.txt'
    ]},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ]
)