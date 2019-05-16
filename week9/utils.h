#pragma once

#include <time.h>


// a small wrapper for convenient time measurements

typedef double timestamp;

timestamp now() {
    struct timespec spec;
    timespec_get(&spec, TIME_UTC);
    return spec.tv_sec + spec.tv_nsec / (1e9);
}

int* prefix_sum (int *array, int N){
	for (int i = 1; i < N; i++) {
		array[i] += array[i-1];
	}
	return array;
}
	
int check (int* pre_data, int* post_data, int N) {
	int check = 1;
	int* prefix_check_data = prefix_sum (pre_data, N);
	for (int i = 0; i < N; i++) {
		if(pre_data[i] != post_data[i]) {
			check = 0;
		}
	}
	return check;
}
