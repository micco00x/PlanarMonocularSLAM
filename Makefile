CXX := g++
CXXOPTS := -std=c++11 -I /usr/local/include/eigen3
BINDIR := bin

.phony:	clean all

all: main

main: main.cpp
	@mkdir -p $(BINDIR)
	$(CXX) $^ $(CXXOPTS) -o $(BINDIR)/$@

clean:
	rm *.o
