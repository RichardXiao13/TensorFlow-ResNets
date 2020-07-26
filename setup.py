import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='tf2-resnets',
    version='0.1.1',
    author='Richard Xiao',
    author_email='richard.xiao13@gmail.com',
    description='ResNet variations for TensorFlow.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/RichardXiao13/TensorFlow-ResNets',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
