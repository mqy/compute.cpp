CXXFLAGS = -O3 -Wall --std c++11 #-fsanitize=address
LDFLAGS  = -I .

OPTIONS += -DNDEBUG

#OPTIONS += -DWORKER_WAIT
#OPTIONS += -DSCHEDULER_WAIT

OPTIONS += -DSPIN_NOP
#OPTIONS += -DSPIN_MEM_PAUSE
OPTIONS += -DSPIN_YIELD

# OPTIONS="-DWORKER_WAIT -DSPIN_MEM_PAUSE" make test-sched
test-sched: test-sched.cpp
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o ./bin/$@ $(LDFLAGS) $(OPTIONS)
	time ./bin/test-sched

clean:
	rm -f *.o ./bin/*

.PHONY: clean test-sched
