#include <math.h>
#include <vector>
#include <cufft.h>

static constexpr size_t threads_per_block = 512;

#define NRANK_2D 2
// XXX TODO: Use cufftPlanMany() in place of cufftPlan2d() to allow for parellel iFFTs
// https://docs.nvidia.com/cuda/cufft/index.html#cufft-code-examples
template <typename B>
__global__ void
domain_iFFT_single_batch(var *domain, int *ax_Len, int *ay_Len) 
{
    // FFT init types
    cufftHandle plan;
    cufftComplex *idata, *odata;
    cufftResult result;
    int NX = *ax_Len;
    int NY = *ay_Len;
    //int n[NRANK_2D] = {NX, NY};
    
    // GPU allocation and copy domain from CPU into idata / odata
    int input_mem_size = sizeof(cufftComplex) * NX * NY;
    size_t host_orig_pitch = NX * sizeof(cufftComplex);
    size_t pitch;

    cudaMallocPitch(
        &domain, 
        &pitch, 
        NX * sizeof(cufftComplex),  // XXX: sizeof(cufftComplex) may need a custom typedef?
        NY
    );

    /*
    cudaMemcpy2D(
        void* dst,                  // Destination memory address
        size_t dpitch,              // Pitch of destination memory
        const void* src,            // Source memory address
        size_t spitch,              // Pitch of source memory
        size_t width,               // Width of matrix transfer (columns in bytes)
        size_t height,              // Height of matrix transfer (rows)
        enum cudaMemcpyKind kind    // Type of transfer
    );
    */

    cudaMemcpy2D(&idata, pitch, domain, host_orig_pitch, NX* sizeof(cufftComplex), NY, cudaMemcpyHostToDevice);
    cudaMemcpy2D(&odata, pitch, domain, host_orig_pitch, NX* sizeof(cufftComplex), NY, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&idata, input_mem_size);
    cudaMalloc((void **)&odata, input_mem_size);

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
    result = cufftExecC2C(plan, idata, odata, CUFFT_INVERSE);
    if (result != CUFFT_SUCCESS) {
        fprintf(stderr, "Cuda error: cufftExecC2C failed"); // Transformers: "More than meets the eye"
        return;
    }

    // Copy device result to host
    cudaMemcpy2D(
        domain, 
        host_orig_pitch, 
        odata, 
        pitch, 
        NX* sizeof(cufftComplex), 
        NY, 
        cudaMemcpyDeviceToHost
    );

    // Clean up
    cufftDestroy(plan);
    cudaFree(idata);
    cudaFree(odata);
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
