python_bin := `which python3`

install:
	{{python_bin}} -m pip install build
	
build:
	{{python_bin}} -m build

publish: build
	{{python_bin}} -m twine upload  dist/*
