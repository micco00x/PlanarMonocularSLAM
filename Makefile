CXX := g++
CXXOPTS := -std=c++11 -Wall -Ofast -I /usr/local/include/eigen3
LDOPTS=-lpthread
BINDIR := bin
BUILDDIR := build

.phony:	clean all

all: main

main: $(BUILDDIR)/mcl/utils.o $(BUILDDIR)/main.o
	@mkdir -p $(BINDIR)
	$(CXX) $^ $(LDOPTS) -o $(BINDIR)/$@

$(BUILDDIR)/main.o: main.cpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $^ $(CXXOPTS) -c -o $@

$(BUILDDIR)/mcl/utils.o: mcl/utils.cpp
	@mkdir -p $(BUILDDIR)/mcl
	$(CXX) $^ $(CXXOPTS) -c -o $@

clean:
	rm $(BUILDDIR)/*.o
	rm $(BUILDDIR)/mcl/*.o
