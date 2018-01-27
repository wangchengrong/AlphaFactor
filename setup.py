from setuptools import setup, find_packages

setup(
    name='posign',
    version='0.01',
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
    author='posignal',
    author_email='poloxue123@163.com',
    description='The signal factor analyze',
    license='MIT',
    keywords='signal factor quantile',
    url='https://github.com/poloxue/posign',

    package_data={
        '': ['*.txt']
    }
)
