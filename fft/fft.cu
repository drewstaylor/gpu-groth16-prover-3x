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

template<typename B>
class CudaVector {
private:
    T* m_begin;
    T* m_end;

    size_t capacity;
    size_t length;
    
    __device__ void expand() {
        capacity *= 2;
        size_t tempLength = (m_end - m_begin);
        T* tempBegin = new T[capacity];

        memcpy(tempBegin, m_begin, tempLength * sizeof(T));
        delete[] m_begin;
        m_begin = tempBegin;
        m_end = m_begin + tempLength;
        length = static_cast<size_t>(m_end - m_begin);
    }
public:
    __device__  explicit CudaVector() : length(0), capacity(16) {
        m_begin = new T[capacity];
        m_end = m_begin;
    }

    __device__ T& operator[] (unsigned int index) {
        return *(m_begin + index);
    }

    __device__ T* begin() {
        return m_begin;
    }
    
    __device__ T* end() {
        return m_end;
    }

    __device__ ~CudaVector() {
        delete[] m_begin;
        m_begin = nullptr;
    }

    __device__ void add(T t) {

        if ((m_end - m_begin) >= capacity) {
            expand();
        }

        new (m_end) T(t);
        m_end++;
        length++;
    }

    __device__ T pop() {
        T endElement = (*m_end);
        delete m_end;
        m_end--;
        return endElement;
    }

    __device__ size_t getSize() {
        return length;
    }
};

//here
//static constexpr size_t threads_per_block = 1024;
static constexpr size_t threads_per_block = 512;

template <typename B>
__global__ void
domain_iFFT(var *domain, const var *a)
{
    //cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);
    B::CudaVector &data = *a->begin;
    *domain->data->iFFT(&data);
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