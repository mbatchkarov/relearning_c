# on mac, brew install libgsl; on raspbian, sudo apt-get install libgsl0-dev
compile:
	gcc -g -I/usr/local/include -I/usr/include -Wall cmeans.c -o exe -lgsl -O3 -lgslcblas -lm && time ./exe
	#gcc -Wall -I/usr/include -c cmeans.c
	#gcc -L/usr/include cmeans.o -lgsl -lgslcblas -lmi

profile:
	valgrind --tool=callgrind ./exe


clean:
	rm -rf *.out *.dSYM exe

# apt-get update && apt-get install -y libgsl-dev && gsl-config --version
docker:
	docker run -it -v `pwd`:/code ubuntu

check:
	cppcheck *.c