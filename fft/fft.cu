#include <math.h>
#include <vector>
#include <cufft.h>

static constexpr size_t threads_per_block = 512;

#define NRANK_2D 2
// XXX TODO: Add a cufftPlanMany() fn to allow for parellel iFFTs
template <typename B>
__global__ void
domain_iFFT_single_batch(var *domain, int *ax_Len, int *ay_Len, const var *aX, const var *aY) 
{
    // FFT data type (init)
    cufftHandle plan;
    cufftComplex *data;
    cufftResult result;
    int NX = *ax_Len;
    int NY = *ay_Len;
    int n[NRANK_2D] = {NX, NY};
    
    int input_mem_size = sizeof(cufftComplex) * NX * NY;

    // XXX TODO: copy host domain into cufftComplex *data
    //here
    /*
    // allocate memory on device and copy h_input into d_array
    Complex     *d_array;
    size_t      host_orig_pitch = N2 * sizeof(Complex);
    size_t      pitch;

    cudaMallocPitch(&d_array, &pitch, N2 * sizeof(Complex), M2);

    cudaMemcpy2D(d_array, pitch, h_input[0], host_orig_pitch, 
        N2* sizeof(Complex), M2, cudaMemcpyHostToDevice);
    */

    // Memory allocation
    cudaMalloc((void **)&data, input_mem_size);

    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Cuda error: Failed to allocate\n");
        return;
    }

    // FFT plan creation
    if (cufftPlan2d(&plan, NX, NY, CUFFT_C2C) != CUFFT_SUCCESS) {
        fprintf(stderr, "Cuda error: Plan creation failed");
        return;
    }

    // FFT execution
    result = cufftExecC2C(plan, data, data, CUFFT_INVERSE);
    if (result != CUFFT_SUCCESS) {
        fprintf(stderr, "Cuda error: ExecC2C failed");
        return;
    }

    // XXX TODO: copy device result to host like: cudaMemcpy2D(host_memory_address, host_destination_pitch, data, pitch, N2* sizeof(Complex), M2, cudaMemcpyDeviceToHost);
    //here

    // Clean up
    cufftDestroy(plan);
    cudaFree(data);
}

/*
template <typename B>
__global__ void
domain_cosetFFT(var *domain, const var *a)
{
    // XXX TODO: write / convert multiplicative_generator
    //domain->data->cosetFFT(*a->data, Fr<mnt4753_pp>::multiplicative_generator);
}

template <typename B>
__global__ void
domain_icosetFFT(var *domain, const var *a)
{
    // XXX TODO: write / convert multiplicative_generator
    //domain->data->icosetFFT(*a->data, Fr<mnt4753_pp>::multiplicative_generator);
}
*/