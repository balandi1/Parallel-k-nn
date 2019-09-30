all: k-nn

k-nn: Knn.cpp KdNode.hpp FileReader.hpp
	g++ -std=c++11 -pthread -o k-nn Knn.cpp

clean:
	rm -f  *o k-nn