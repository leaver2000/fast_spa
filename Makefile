
clean:
	rm -rf build/ dist/ src/*.egg-info /src/**/*.pyc src/**/*.so fastspa/*.so fastspa/*.c

in-place:
	python setup.py build_ext --inplace && pytest tests -s