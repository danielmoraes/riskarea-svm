CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 2
OS = $(shell uname)

all: clean svm-train svm-predict svm-scale lib

lib: svm.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libsvm.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,libsvm.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} bin/svm.o -o bin/libsvm.so.$(SHVER)

svm-predict: svm-predict.c svm.o
	$(CXX) $(CFLAGS) svm-predict.c bin/svm.o -o bin/svm-predict -lm
svm-train: svm-train.c svm.o
	$(CXX) $(CFLAGS) svm-train.c bin/svm.o -o bin/svm-train -lm
svm-scale: svm-scale.c
	$(CXX) $(CFLAGS) svm-scale.c -o bin/svm-scale
svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c -o bin/svm.o svm.cpp
clean:
	rm -f *~ bin/svm.o bin/svm-predict bin/svm-train bin/svm-scale bin/libsvm.so.$(SHVER)
