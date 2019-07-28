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

static constexpr size_t threads_per_block = 512;

#define NRANK_2D 2

template <typename B>
__global__ void
domain_iFFT_single_batch(const int *ax_Len, const int *ay_Len, const var *aX, const var *aY) 
{
    // FFT data type (init)
    cufftHandle plan;
    cufftComplex *data;
    int NX = ax_Len;
    int NY = ay_Len;
    int n[NRANK_2D] = {NX, NY};
    
    int input_mem_size = sizeof(cufftComplex) * NX * NY;

    // Memory allocation
    cudaMalloc((void **)&data, input_mem_size);

    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Cuda error: Failed to allocate\n");
        return;
    }

    // FFT plan creation
    if (cufftPlan2d(&plan, NX, CUFFT_C2C) != CUFFT_SUCCESS) {
        fprintf(stderr, "Cuda error: Plan creation failed");
        return;
    }

    // FFT execution
    if (cufftExecC2C(plan, data, data, CUFFT_INVERSE)) {
        fprintf(stderr, "Cuda error: ExecC2C failed");
        return;
    }

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