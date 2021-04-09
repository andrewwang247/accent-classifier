#!/bin/bash

# Copyright 2020 Siwei Wang.
# Run static checks on all Python code.

set -Euo pipefail

# List of all Python files.
pyfiles="*.py"

printf "Running autopep...\n\n"
autopep8 -i -a -a $pyfiles

printf "Running pylint...\n\n"
pylint $pyfiles

printf "Running pycodestyle...\n\n"
pycodestyle $pyfiles

printf "\nRunning pydocstyle...\n\n"
pydocstyle $pyfiles

printf "\nRunning mypy...\n\n"
mypy $pyfiles
