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

/**
 * An evaluation domain.
 */
template<typename FieldT, T>
class evaluation_domain {
public:

    const size_t m;
 
    /**
     * Construct an evaluation domain S of size m, if possible.
     *
     * (See the function get_evaluation_domain below.)
     */
    evaluation_domain(const size_t m) : m(m) {};

    /**
     * Get the idx-th element in S.
     */
    //virtual FieldT get_domain_element(const size_t idx) = 0;
 
    /**
     * Compute the FFT, over the domain S, of the vector a.
     */
    //virtual void FFT(T::CudaVector<FieldT> &a) = 0;

    /**
     * Compute the inverse FFT, over the domain S, of the vector a.
     */
    virtual void iFFT(T::CudaVector<FieldT> &a) = 0;

    /**
     * Compute the FFT, over the domain g*S, of the vector a.
     */
    //virtual void cosetFFT(T::CudaVector<FieldT> &a, const FieldT &g) = 0;

    /**
     * Compute the inverse FFT, over the domain g*S, of the vector a.
     */
    //virtual void icosetFFT(T::CudaVector<FieldT> &a, const FieldT &g) = 0;

    /**
     * Evaluate all Lagrange polynomials.
     *
     * The inputs are:
     * - an integer m
     * - an element t
     * The output is a vector (b_{0},...,b_{m-1})
     * where b_{i} is the evaluation of L_{i,S}(z) at z = t.
    */
    //virtual std::vector<FieldT> evaluate_all_lagrange_polynomials(const FieldT &t) = 0;
    //virtual CudaVector<FieldT> evaluate_all_lagrange_polynomials(const FieldT &t) = 0;

    /**
     * Evaluate the vanishing polynomial of S at the field element t.
    */
    //virtual FieldT compute_vanishing_polynomial(const FieldT &t) = 0;
 
    /**
     * Add the coefficients of the vanishing polynomial of S to the coefficients of the polynomial H.
     */
    //virtual void add_poly_Z(const FieldT &coeff, CudaVector<FieldT> &H) = 0;

    /**
     * Multiply by the evaluation, on a coset of S, of the inverse of the vanishing polynomial of S.
     */
    //virtual void divide_by_Z_on_coset(CudaVector<FieldT> &P) = 0;
 };

 template<typename T>
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
domain_iFFT(cudaStream_t &strm, FieldT::evaluation_domain var *domain, const var *a)
{
    cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);
    T::CudaVector<FieldT> &data = *a->data;
    domain->data->iFFT(data);
}

template <typename B>
__global__ void
domain_cosetFFT(FieldT::evaluation_domain var *domain, T::CudaVector<FieldT> var *a)
{
    // XXX TODO: write / convert multiplicative_generator
    //domain->data->cosetFFT(*a->data, Fr<mnt4753_pp>::multiplicative_generator);
}

template <typename B>
__global__ void
domain_icosetFFT(FieldT::evaluation_domain var *domain, T::CudaVector<FieldT> var *a)
{
    // XXX TODO: write / convert multiplicative_generator
    //domain->data->icosetFFT(*a->data, Fr<mnt4753_pp>::multiplicative_generator);
}