build:
	/opt/homebrew/opt/python@3.11/bin/python3.11 -m build

publish:
	/opt/homebrew/opt/python@3.11/bin/python3.11 -m twine upload  dist/*
