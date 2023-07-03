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

# example: ACCELERATE=1 make ./test_blas
test-blas:
	gcc $(CFLAGS) $(LDFLAGS) test_blas.c -o test_blas

run-tests:
	mkdir -p out
	# ACCELERATE=1 make
	# ./test_blas > ./out/accelerate.txt

	# OPENBLAS=1 make
	# ./test_blas > ./out/openblas.txt

	BLIS=1 make
	./test_blas > ./out/blis.txt
