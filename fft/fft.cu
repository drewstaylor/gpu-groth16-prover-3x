#include <math.h>
#include <vector>
#include <cufft.h>

/*
B::domain_iFFT(domain, ca);
B::domain_iFFT(domain, cb);

B::domain_cosetFFT(domain, ca);
B::domain_cosetFFT(domain, cb);

B::domain_iFFT(domain, cc);
B::domain_cosetFFT(domain, cc);

B::domain_icosetFFT(domain, H_tmp);



void mnt4753_libsnark::domain_iFFT(mnt4753_libsnark::evaluation_domain *domain,
                                   mnt4753_libsnark::vector_Fr *a) {
  T::CudaVector<Fr<mnt4753_pp>> &data = *a->data;
  domain->data->iFFT(data);
}
void mnt4753_libsnark::domain_cosetFFT(
    mnt4753_libsnark::evaluation_domain *domain,
    mnt4753_libsnark::vector_Fr *a) {
  domain->data->cosetFFT(*a->data, Fr<mnt4753_pp>::multiplicative_generator);
}
void mnt4753_libsnark::domain_icosetFFT(
    mnt4753_libsnark::evaluation_domain *domain,
    mnt4753_libsnark::vector_Fr *a) {
  domain->data->icosetFFT(*a->data, Fr<mnt4753_pp>::multiplicative_generator);
}

*/

// B::domain_iFFT(a, b);
// B::domain_cosetFFT(a, ca);
// B::domain_icosetFFT(a, b);

//static constexpr size_t threads_per_block = 1024;
static constexpr size_t threads_per_block = 512;

#define NX 256; // Size of transform 
#define BATCH 10; // The number of transforms to do of size NX

cufftHandle plan;

template <typename B>
__global__ void
domain_iFFT(var *domain, var *a) 
{
    // FFT data type
    //cufftComplex *data = a;
    cufftComplex *data;
    int input_mem_size = sizeof(cufftComplex) * NX * BATCH;

    // Memory allocation
    //cudaMalloc((void**)&data, sizeof(cufftComplex)*NX*BATCH);
    cudaMalloc((void **)&data, input_mem_size);

    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Cuda error: Failed to allocate\n");
        return;
    }

    // FFT plan creation
    if (cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH) != CUFFT_SUCCESS) {
        fprintf(stderr, "Cuda error: Plan creation failed");
        return;
    }

    // FFT execution
    if (cufftExecC2C(plan, data, data, CUFFT_INVERSE)) {
        fprintf(stderr, "Cuda error: ExecC2C failed");
        return;
    }

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