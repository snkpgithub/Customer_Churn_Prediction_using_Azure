from setuptools import setup, find_packages

setup(
    name='customer_churn_prediction',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'azureml-core',
        'azureml-train-automl-runtime',
        'matplotlib',
        'seaborn',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for predicting customer churn using Azure Machine Learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/customer-churn-prediction',
)
