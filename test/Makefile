# PYTHON_VERSION = 3.8
# PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

# # location of the Boost Python include files and library

# BOOST_INC = /usr/include
# BOOST_LIB = /lib

# # compile mesh classes
# TARGET = hello

# $(TARGET).so: $(TARGET).o
# 	# g++ -shared -Wl,--export-dynamic $(TARGET).o -L$(BOOST_LIB) -lboost_python-$(PYTHON_VERSION) -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o $(TARGET).so
# 	g++ -shared -Wl,--export-dynamic $(TARGET).o -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o $(TARGET).so

# $(TARGET).o: $(TARGET).cpp
# 	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c $(TARGET).cpp

CC = g++
PYLIBPATH = $(shell python-config --exec-prefix)/lib
LIB = -L$(PYLIBPATH) $(shell python-config --libs) -lboost_python
OPTS = $(shell python-config --includes) -O2

default: hello.so
	@python ./hello.py

hello.so: hello.o
	$(CC) $(LIB) -Wl,-rpath,$(PYLIBPATH) -shared $< -o $@

hello.o: hello.cpp Makefile
	$(CC) $(OPTS) -c $< -o $@

clean:
	rm -rf *.so *.o

.PHONY: default clean