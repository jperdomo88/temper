from setuptools import setup, find_packages

setup(
    name="crucible",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],  # No external dependencies
    entry_points={
        'console_scripts': [
            'temper-validate=crucible.run_validation:main',
        ],
    },
)
