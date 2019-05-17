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
	int sum_arr[N];
	for (int i = 0; i < N; i++) {
		if (i == 0) sum_arr[i] = 0;
		else sum_arr[i] = sum_arr[i-1] + array[i-1];
		//printf("N = %d\telem = %d\tpre = %d\n", i+1, array[i], sum_arr[i]);

	}
	array = sum_arr;
	return array;
}
	
int check (int* pre_data, int* post_data, int N) {
	int check = 1;
	int* prefix_check_data = prefix_sum (pre_data, N);
	for (int i = 0; i < N; i++) {
		if(prefix_check_data[i] != post_data[i]) {
			check = 0;
		}
	}
	return check;
}

