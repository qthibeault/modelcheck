# ModelCheck

This program computes the safe region and estimates the robustness of a
classifier with respect to an input.

## Dependencies
This program is configured to run with Python 3.10

## Usage
This program uses [Pipenv](https://pipenv.pypa.io) for reproducible
environments. To get started, run the command `pipenv install` to create the
virtual environment. Then, you can test an image using the command 
`pipenv run modelcheck images/n01644373_tree_frog.JPEG`. For more information
about the program options available, use the command
`pipenv run modelcheck --help`.
