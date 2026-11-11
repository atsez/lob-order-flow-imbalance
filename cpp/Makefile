CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -pedantic

ofi_engine: ofi_engine.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f ofi_engine
