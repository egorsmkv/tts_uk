python_bin := `which python3.11`

build:
	{{python_bin}} -m build

publish: build
	{{python_bin}} -m twine upload  dist/*
