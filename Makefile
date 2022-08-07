.PHONY: clean
clean:
	rm -rf build/

.PHONY: build
build:
	python -m build


clean-build: clean build

.PHONY: publish
publish:
	python -m twine upload --skip-existing --verbose dist/*
