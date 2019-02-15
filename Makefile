CXX := g++
CXXOPTS := -std=c++11 -Wall -Ofast -I /usr/local/include/eigen3
LDOPTS=-lpthread
BINDIR := bin

.phony:	clean all

all: main

main: main.cpp
	@mkdir -p $(BINDIR)
	$(CXX) $^ $(CXXOPTS) $(LDOPTS) -o $(BINDIR)/$@

clean:
	rm *.o
