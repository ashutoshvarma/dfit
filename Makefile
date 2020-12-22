PYTHON?=python3
PYTESTFLAGS=
SETUPFLAGS=

ifeq ($(OS),Windows_NT) 
    detected_OS := Windows
else
    detected_OS := $(shell sh -c 'uname 2>/dev/null || echo Unknown')
endif

ifeq ($(detected_OS), Windows)
    _CFLAGS += /Od /Zi
endif
ifeq ($(detected_OS), Linux)
    _CFLAGS += -O0 -g3
endif


.PHONY: all inplace inplace2 rebuild-sdist sdist build require-cython wheel_manylinux wheel

.PHONY: all
all: inplace

# Build in-place
.PHONY: inplace
inplace:
	CFLAGS='$(_CFLAGS)' $(PYTHON) setup.py $(SETUPFLAGS) build_ext -i --with-signature --with-coverage --with-clines --cython-gdb

.PHONY: gdb
gdb: inplace
	gdb -return-child-result -batch -ex r -ex bt --args $(PYTHON) -m pytest $(PYTESTFLAGS)

.PHONY: sdist
sdist:
	rm -f dist/*.tar.gz
	find src -name '*.cpp' -exec rm -f {} \;
	$(PYTHON) setup.py $(SETUPFLAGS) sdist

.PHONY: wheel
wheel:
	$(PYTHON) setup.py $(SETUPFLAGS) bdist_wheel

.PHONY: test
test: inplace
	$(PYTHON) -m pytest $(TESTFLAGS) $(TESTOPTS) 

.PHONY: test_wheel
test_wheel: wheel	
	pip install -U dist/*.whl
	$(PYTHON) -m pytest $(PYTESTFLAGS)


.PHONY: valgrind_all
valgrind_all: inplace
	# Don't know why but supression file is not supressing any python malloc errors
	# So for Python >= 3.6, using this hack. 
	#valgrind --tool=memcheck --leak-check=full  --suppressions=valgrind-python.supp
	PYTHONMALLOC=malloc valgrind --leak-check=full \
		         				--show-leak-kinds=all \
				         		--track-origins=yes \
						        --verbose \
								--log-file=valgrind-out.log \
								--suppressions=valgrind-python.supp \
								$(PYTHON) -m pytest $(PYTESTFLAGS) 

.PHONY: valgrind
valgrind: inplace
	PYTHONMALLOC=malloc valgrind --leak-check=full \
								--show-leak-kinds=definite 	\
								--errors-for-leak-kinds=definite \
								--suppressions=valgrind-python.supp \
								$(PYTHON) -m pytest $(PYTESTFLAGS) 

.PHONY: clean
clean:
	find . -path ./venv -prune -o \( -name '*.o' -o -name '*.so' -o -name '*.py[cod]' -o -name '*.dll' -o -name '*.cpp' \) -exec rm -f {} \;
	rm -rf build

.PHONY: realclean
realclean: clean 
	rm -f TAGS
	$(PYTHON) setup.py clean -a
