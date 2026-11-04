@echo off
REM Rebuild the HTML documentation from the docstrings and save the test report.
REM Run from the code folder: make_docs.bat
pdoc load_env.py gaps.py fill_methods.py experiment.py optimise.py -d google -o docs
python -m pytest tests -v > results\test_report.txt
echo Docs written to docs\index.html and test report to results\test_report.txt
