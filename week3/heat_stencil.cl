
__kernel void heat_stencil(
    __global float* b, 
    __global const float* a, 
    int N
) {
    // obtain position of this 'thread'
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);

    // if beyond boundaries => skip this one
    if (i >= N || j >= N) return;
    if (i == N/4 && j == N/4) {
	b[i*N+j] = a[i*N+j];
	return;
     }
	float tl = a[i*N+j];
	float tr = a[i*N+j];
	float tu = a[i*N+j];
	float td = a[i*N+j];
	if(j!=0){
		tl=a[i*N+(j-1)];
	}
	if(j!=N-1){
		tr=a[i*N+(j+1)];
	}	
	if(i!=0){
		tu=a[(i-1)*N+j];
	}	
	if(i!=N-1){
		td=a[(i+1)*N+j];
	}	

    b[i*N+j] = a[i*N+j] +0.2*(tl+tr+tu+td+(-4.0f*a[i*N+j]));
}
