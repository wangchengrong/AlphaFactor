from setuptools import setup, find_packages

setup(
    name='AlphaFactor',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'tushare'
    ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'],

    # metadata for upload to PyPI
    author='posign',
    author_email='poloxue123@163.com',
    description='multi factor analyze',
    license='MIT',
    keywords='Alpha Factor Quantile',
    url='https://github.com/poloxue/AlphaFactor',

    package_data={
        '': ['*.txt']
    }
)
