#include <cstdint>
#include <vector>
#include <chrono>
#include <memory>
#include <math.h>
#include <cooperative_groups.h>

#include "curves.cu"
#include "sharedmem.cuh"

namespace cg = cooperative_groups;

// C is the size of the precomputation
// R is the number of points we're handling per thread
template< typename EC, int C = 4, int RR = 16 >
__global__ void
ec_multiexp_straus(var *out, const var *multiples_, const var *scalars_, size_t N)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    size_t n = (N + RR - 1) / RR;
    if (idx < n) {
        size_t R = (idx < n - 1) ? RR : (N % RR);

        typedef typename EC::group_type Fr;
        static constexpr int JAC_POINT_LIMBS = 3 * EC::field_type::DEGREE * ELT_LIMBS;
        static constexpr int AFF_POINT_LIMBS = 2 * EC::field_type::DEGREE * ELT_LIMBS;
        int out_off = idx * JAC_POINT_LIMBS;
        int m_off = idx * RR * AFF_POINT_LIMBS;
        int s_off = idx * RR * ELT_LIMBS;

        Fr scalars[RR];
        for (int j = 0; j < R; ++j) {
            Fr::load(scalars[j], scalars_ + s_off + j*ELT_LIMBS);
            Fr::from_monty(scalars[j], scalars[j]);
        }

        const var *multiples = multiples_ + m_off;

        // i is smallest multiple of C such that i > 753
        int i = C * ((753 + C - 1) / C); // C * ceiling(753/C)
        assert((i - C * 753) < C);
        static constexpr var C_MASK = (1U << C) - 1U;

        EC x;
        EC::set_zero(x);
        while (i >= C) {
            EC::mul_2exp<C>(x, x);
            i -= C;

            int q = i / digit::BITS, r = i % digit::BITS;
            for (int j = 0; j < R; ++j) {
                auto g = fixnum::layout();
                var s = g.shfl(scalars[j].a, q);
                var win = (s >> r) & C_MASK;
                // Handle case where C doesn't divide digit::BITS
                int bottom_bits = digit::BITS - r;
                // detect when window overlaps digit boundary
                if (bottom_bits < C) {

                    s = g.shfl(scalars[j].a, q + 1);

                    win |= (s << bottom_bits) & C_MASK;
                }
                if (win > 0) {
                    EC m;
                    EC::load_affine(m, multiples + ((win-1)*N + j)*AFF_POINT_LIMBS);
                    EC::mixed_add(x, x, m);
                }
            }
        }
        EC::store_jac(out + out_off, x);
    }
}

template< typename EC >
__global__ void
ec_multiexp(var *X, const var *W, size_t n)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    if (idx < n) {
        typedef typename EC::group_type Fr;
        EC x;
        Fr w;
        int x_off = idx * EC::NELTS * ELT_LIMBS;
        int w_off = idx * ELT_LIMBS;

        EC::load_affine(x, X + x_off);
        Fr::load(w, W + w_off);

        // We're given W in Monty form for some reason, so undo that.
        Fr::from_monty(w, w);
        EC::mul(x, w.a, x);

        EC::store_jac(X + x_off, x);
    }
}

template< typename EC >
__global__ void
ec_sum_all(var *X, const var *Y, size_t n)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    if (idx < n) {
        EC z, x, y;
        int off = idx * EC::NELTS * ELT_LIMBS;

        EC::load_jac(x, X + off);
        EC::load_jac(y, Y + off);

        EC::add(z, x, y);

        EC::store_jac(X + off, z);
    }
}

template <typename EC>
__inline__ __device__
EC shfl_down(EC x, int offset){

    var tmp[EC::NELTS * ELT_LIMBS];
    var res[EC::NELTS * ELT_LIMBS];
    EC result;

    EC::store_jac(tmp, x);
    #pragma unroll
    for(int i = 0; i <  EC::NELTS * ELT_LIMBS; i++)
        res[i] = __shfl_down_sync(__activemask(), tmp[i], offset);

    EC::load_jac(result, res);
    return result;
}


// Reduce using Brent's theorem
template <typename EC, size_t blockSize>
__global__ void
ec_sum_all_brent(var *resOut, const var *resIn, size_t n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

    // Shared mem size is determined by the host app at run time
    SharedMemory<EC> smem;
    EC* sdata = smem.getPointer();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
  int elts_per_block = D / BIG_WIDTH;
  int tileIdx = T / BIG_WIDTH;

  i = elts_per_block * B + tileIdx;

  EC sum;
  EC::set_zero(sum);

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
      EC x;
      EC::load_jac(x, resIn + (i * EC::NELTS * ELT_LIMBS));
      EC::add(sum, x, sum);

      // ensure we don't read out of bounds -- this is optimized away for powerOf2
      // sized arrays
      if (i + blockSize < n) {
          EC::load_jac(x, resIn + ((i + blockSize) * EC::NELTS * ELT_LIMBS));
          EC::add(sum, x, sum);
      }

      i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = sum;
  cg::sync(cta);

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
      EC::add(sum, sum, sdata[tid + 256]);
      sdata[tid] = sum;
  }

  cg::sync(cta);

  if ((blockSize >= 256) && (tid < 128)) {
    EC::add(sum, sum, sdata[tid + 128]);
    sdata[tid] = sum;
  }

  cg::sync(cta);

  if ((blockSize >= 128) && (tid < 64)) {
      EC::add(sum, sum, sdata[tid + 64]);
      sdata[tid] = sum;
  }

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64) {
        EC::add(sum, sum, sdata[tid + 32]);
    }
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
        EC y = shfl_down<EC>(sum, offset);
        EC::add(sum, sum, y);
        // mySum += tile32.shfl_down(mySum, offset);
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0) {
      EC::store_jac(resOut + (blockIdx.x * EC::NELTS * ELT_LIMBS), sum);
      // g_odata[blockIdx.x] = mySum;
  }
}




//static constexpr size_t threads_per_block = 1024;
static constexpr size_t threads_per_block = 512;

template< typename EC, int C, int R >
void
ec_reduce(cudaStream_t &strm, var *out, const var *multiples, const var *scalars, size_t N) //here
{
    cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);

    size_t n = (N + R - 1) / R;

    size_t nblocks = (n * BIG_WIDTH + threads_per_block - 1) / threads_per_block;

    ec_multiexp_straus<EC, C, R><<< nblocks, threads_per_block, 0, strm>>>(out, multiples, scalars, N);



    static constexpr size_t pt_limbs = EC::NELTS * ELT_LIMBS;
    size_t r = n & 1, m = n / 2;

    for ( ; m != 0; r = m & 1, m >>= 1) {
        nblocks = (m * BIG_WIDTH + threads_per_block - 1) / threads_per_block;

        ec_sum_all<EC><<<nblocks, threads_per_block, 0, strm>>>(out, out + m*pt_limbs, m);
        if (r)
            ec_sum_all<EC><<<1, threads_per_block, 0, strm>>>(out, out + 2*m*pt_limbs, 1);
    }


    // dim3 dimBlock(threads_per_block, 1, 1);
    // dim3 dimGrid(nblocks, 1, 1);
    // ec_sum_all_brent<EC, threads_per_block><<<dimGrid, dimBlock, 32, strm>>>(out, out, n);
}

static inline double as_mebibytes(size_t n) {
    return n / (long double)(1UL << 20);
}

void print_meminfo(size_t allocated) {
    size_t free_mem, dev_mem;
    cudaMemGetInfo(&free_mem, &dev_mem);
    fprintf(stderr, "Allocated %zu bytes; device has %.1f MiB free (%.1f%%).\n",
            allocated,
            as_mebibytes(free_mem),
            100.0 * free_mem / dev_mem);
}

struct CudaFree {
    void operator()(var *mem) { cudaFree(mem); }
};
typedef std::unique_ptr<var, CudaFree> var_ptr;

var_ptr
allocate_memory(size_t nbytes, int dbg = 0) {
    var *mem = nullptr;
    cudaMallocManaged(&mem, nbytes);
    if (mem == nullptr) {
        fprintf(stderr, "Failed to allocate enough device memory\n");
        abort();
    }
    //if (dbg)
        //print_meminfo(nbytes);
    return var_ptr(mem);
}

var_ptr
load_scalars(size_t n, FILE *inputs)
{
    static constexpr size_t scalar_bytes = ELT_BYTES;
    size_t total_bytes = n * scalar_bytes;

    auto mem = allocate_memory(total_bytes);
    if (fread((void *)mem.get(), total_bytes, 1, inputs) < 1) {
        fprintf(stderr, "Failed to read scalars\n");
        abort();
    }
    return mem;
}

template< typename EC >
var_ptr
load_points(size_t n, FILE *inputs)
{
    typedef typename EC::field_type FF;

    static constexpr size_t coord_bytes = FF::DEGREE * ELT_BYTES;
    static constexpr size_t aff_pt_bytes = 2 * coord_bytes;
    static constexpr size_t jac_pt_bytes = 3 * coord_bytes;

    size_t total_aff_bytes = n * aff_pt_bytes;
    size_t total_jac_bytes = n * jac_pt_bytes;

    auto mem = allocate_memory(total_jac_bytes);
    if (fread((void *)mem.get(), total_aff_bytes, 1, inputs) < 1) {
        fprintf(stderr, "Failed to read all curve poinst\n");
        abort();
    }

    // insert space for z-coordinates
    char *cmem = reinterpret_cast<char *>(mem.get()); //lazy
    for (size_t i = n - 1; i > 0; --i) {
        char tmp_pt[aff_pt_bytes];
        memcpy(tmp_pt, cmem + i * aff_pt_bytes, aff_pt_bytes);
        memcpy(cmem + i * jac_pt_bytes, tmp_pt, aff_pt_bytes);
    }
    return mem;
}

template< typename EC >
var_ptr
load_points_affine(size_t n, FILE *inputs)
{
    typedef typename EC::field_type FF;

    static constexpr size_t coord_bytes = FF::DEGREE * ELT_BYTES;
    static constexpr size_t aff_pt_bytes = 2 * coord_bytes;

    size_t total_aff_bytes = n * aff_pt_bytes;

    auto mem = allocate_memory(total_aff_bytes);
    if (fread((void *)mem.get(), total_aff_bytes, 1, inputs) < 1) {
        fprintf(stderr, "Failed to read all curve poinst\n");
        abort();
    }
    return mem;
}
