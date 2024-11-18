from setuptools import setup, find_packages

setup(
    name="kannada_nlp_project",                # Package name
    version="0.1.0",                           # Package version
    packages=find_packages(),                  # Automatically find the packages
    install_requires=[                         # List of dependencies
        "tensorflow>=2.0",
        "pandas>=1.1.0",
        "matplotlib>=3.0.0",
        "seaborn>=0.10.0",
        "wordcloud>=1.8.1",
        "nltk>=3.5",
        "scikit-learn>=0.24.0",
    ],
    entry_points={                              # Command line scripts (if needed)
        'console_scripts': [
            'train_model=kannada_nlp.train:train_model',
            'preprocess_data=kannada_nlp.preprocess:preprocess_text_data_parallel',
        ],
    },
    author="Your Name",                        # Your name
    author_email="amrithniyogi25@gmail.com",     # Your email
    description="A package for Kannada NLP tasks", # Short description
    long_description=open('README.md').read(), # Long description (from README.md)
    long_description_content_type="text/markdown",
    url="https://github.com/AmrithNiyogi/kannada-nlp-model", # GitHub URL (replace with your own)
    classifiers=[                              # Categorize the package
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Custom :: Kannada NLP Model License v1.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',                    # Minimum Python version
)
