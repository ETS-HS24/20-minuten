from setuptools import find_packages, setup

setup(
    name='swissdox',
    packages=find_packages(),
    version='1.1.0',
    url='https://github.com/ETS-HS24/20-minuten/tree/main/Swissdox',
    description='An SDK for the Swissdox@LiRI database.',
    author='Nordin Dari',
    author_email='nordinbensalem.dari@uzh.ch',
    install_requires=[
        'requests',
        'pyyaml',
        'python-dotenv',
        'jsonschema'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    include_package_data=True,
    package_data={
        '': ['schemas/*.yaml', 'schemas/*.json'],
    },
)
