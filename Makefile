
clean:
	rm -rf build/ dist/ fast_spa/*.egg-info fast_spa/*.so fast_spa/*.c

inplace:
	python setup.py build_ext --inplace && pytest tests -s