CXXFLAGS += -O3 -Wall --std c++11 #-fsanitize=address
LDFLAGS  += -I .

OPTIONS =
OPTIONS += -DNDEBUG=
OPTIONS += -DWORKER_WAIT
#OPTIONS += -DPRODUCER_WAIT
OPTIONS += -DSPIN_NOP
#OPTIONS += -DSPIN_MEM_PAUSE
OPTIONS += -DSPIN_YIELD

test-ggml-thread: test-ggml-thread.cpp
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(OPTIONS)

clean:
	rm -f *.o test-ggml-thread

test: clean test-ggml-thread
	time ./test-ggml-thread

all: test-ggml-thread
