all:
	rm -rf test_load_graph
	gcc -g -std=c99 load_gragh.c -o test_load_graph  -ltensorflow
clean:
	rm -rf test*
