from setuptools import setup, find_packages

# with open("README.md", 'r') as f:
#     long_description = f.read()


setup(
    name='categorical_embedder',
    version='0.1.0',
    description='transformer for categorical features',
    # long_description=long_description,
    license="MIT",
    author='Vito Stamatti',
    package_dir={'':'.'},
    packages=find_packages(where='.'),
    install_requires=[
        # 'pandas', 
        # 'numpy', 
        'scikit-learn', 
        # 'dill', 
        # 'loguru', 
    ],
),