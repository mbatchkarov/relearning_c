compile:
	gcc -I/usr/local/include -Wall -lgsl -lgslcblas -lm cmeans.c -o exe && time ./exe

check:
	cppcheck *.c