CFLAGS=-O3 -std=c11 #-fsanitize=address
LDFLAGS=

ifdef ACCELERATE
	CFLAGS  +=-DUSE_ACCELERATE
	LDFLAGS = -framework Accelerate
endif

ifdef OPENBLAS
	# homebrew
	CFLAGS += -DUSE_OPENBLAS -I/usr/local/opt/openblas/include
	LDFLAGS = -lopenblas -L/usr/local/opt/openblas/lib
endif

# https://github.com/flame/blis
# ./configure --enable-cblas auto
# make check -j 4
# make install
ifdef BLIS
	CFLAGS += -DUSE_BLIS -I/usr/local/include/blis
	LDFLAGS = -lblis -L/usr/local/lib
endif

ifdef MKL
# . /opt/intel/oneapi/setvars.sh
MKL_ROOT=/opt/intel/oneapi/mkl/latest
CFLAGS += -DUSE_MKL
CFLAGS += $(shell PKG_CONFIG_PATH=$(MKL_ROOT)/lib/pkgconfig pkg-config --cflags mkl-sdl)
LDFLAGS = $(shell PKG_CONFIG_PATH=$(MKL_ROOT)/lib/pkgconfig pkg-config --libs mkl-sdl)

build-test:
	. /opt/intel/oneapi/setvars.sh --force
	gcc $(CFLAGS) $(LDFLAGS) test_blas.c -o test_blas
else
# example: ACCELERATE=1 make
build-test:
	gcc $(CFLAGS) $(LDFLAGS) test_blas.c -o test_blas
endif

run-test:
	mkdir -p out
# 	ACCELERATE
#	./test_blas > ./out/accelerate.txt
# 	OPENBLAS
#	./test_blas > ./out/openblas.txt
# 	BLIS
#	./test_blas > ./out/blis.txt
# 	MKL
	. /opt/intel/oneapi/setvars.sh --force && ./test_blas > ./out/mkl.txt

