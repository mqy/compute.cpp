CXXFLAGS = -O3 -Wall --std c++11 #-fsanitize=address
LDFLAGS  = -I .

OPTIONS = -DNDEBUG

test-sched: test-sched.cpp
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o ./bin/$@ $(LDFLAGS) $(OPTIONS)

run-test-sched: test-sched
	./bin/test-sched

clean:
	rm -f *.o ./bin/*

.PHONY: clean test-sched
