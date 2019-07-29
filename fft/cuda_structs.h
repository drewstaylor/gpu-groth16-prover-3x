#ifndef CUDA_STRUCTS_H
#define CUDA_STRUCTS_H

#include <stdint.h>
#include <assert.h>
#include <type_traits>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#define DEVICE_FUNC __device__
#define HOST_DEVICE_FUNC __host__ __device__ 
#define DEVICE_VAR __device__
#define HOST_DEVICE_VAR __host__ __device__
#define CONST_MEMORY __constant__

#define HALF_N 4
#define N_BASE 8
#define N_DOUBLED 16
#define N_BITLEN 254
#define R_LOG 256
#define BITS_PER_LIMB 32

#define USE_PROJECTIVE_COORDINATES

#define WARP_SIZE 32
#define DEFAUL_NUM_OF_THREADS_PER_BLOCK 256

#define CHECK_BIT(var,pos) ((var) & (1<<(pos)))
#define SET_BIT(var,pos) ((var) |= (1<<(pos)))

struct uint64_g
{
    union 
    {
        uint64_t as_long;
        struct 
        {
            uint32_t low;
            uint32_t high;
        };      
    };
};

struct uint128_g
{
    union
    {
        uint32_t n[4];
        struct
        {
            uint64_t low;
            uint64_t high;
        };
    };
};

struct uint128_with_carry_g
{
    uint128_g val;
    uint32_t carry;
};

// XXX TODO: https://stackoverflow.com/questions/10297067/in-a-cuda-kernel-how-do-i-store-an-array-in-local-thread-memory
struct uint256_g
{
    union
    {
        uint32_t n[8];
        uint64_t nn[4];
        struct
        {
            uint128_g low;
            uint128_g high;
        };
    };
};

struct uint512_g
{
    union
    {
        uint32_t n[16];
        uint64_t nn[8];
        uint256_g l[2];
    };
};

struct ec_point
{
    uint256_g x;
    uint256_g y;
    uint256_g z;
};

struct affine_point
{
    uint256_g x;
    uint256_g y;
};

// This is a field embedded into a group of points on elliptic curve
struct embedded_field
{
	uint256_g rep_;
    
    DEVICE_FUNC explicit embedded_field(const uint256_g rep);
	DEVICE_FUNC embedded_field();
	
	static DEVICE_FUNC embedded_field zero();
	static DEVICE_FUNC embedded_field one();
	
    DEVICE_FUNC bool operator==(const embedded_field& other) const;
	DEVICE_FUNC bool operator!=(const embedded_field& other) const;
	
	DEVICE_FUNC operator uint256_g() const;
	DEVICE_FUNC embedded_field operator-() const;
	
	// For now we assume that highest possible limb bit is zero for the field modulus
	DEVICE_FUNC embedded_field& operator+=(const embedded_field& other);
	DEVICE_FUNC embedded_field& operator-=(const embedded_field& other);
	
	// Here we mean montgomery multiplication
	DEVICE_FUNC embedded_field& operator*=(const embedded_field& other);
	
	friend DEVICE_FUNC embedded_field operator+(const embedded_field& left, const embedded_field& right);
	friend DEVICE_FUNC embedded_field operator-(const embedded_field& left, const embedded_field& right);
	friend DEVICE_FUNC embedded_field operator*(const embedded_field& left, const embedded_field& right);
};
	
DEVICE_FUNC embedded_field operator+(const embedded_field& left, const embedded_field& right);
DEVICE_FUNC embedded_field operator-(const embedded_field& left, const embedded_field& right);
DEVICE_FUNC embedded_field operator*(const embedded_field& left, const embedded_field& right);


// Miscellaneous helpful staff :)
DEVICE_FUNC inline bool get_bit(const uint256_g& x, uint32_t index)
{
	auto num = x.n[index / 32];
	auto pos = index % 32;
	return CHECK_BIT(num, pos);
}

DEVICE_FUNC inline void set_bit(uint256_g& x, uint32_t index)
{
	auto& num = x.n[index / 32];
	auto pos = index % 32;
    num |= (1 << pos);
	//SET_BIT(num, pos);
}

// Init this badboy
bool CUDA_init();
void get_device_info();


#ifdef __CUDACC__

// Some global constants
// XXX TODO: it's better to embed this constants at compile time rather than taking them from constant memory
// seems like a way for later optimization

// Curve order field
extern DEVICE_VAR CONST_MEMORY uint256_g EMBEDDED_FIELD_P; 
extern DEVICE_VAR CONST_MEMORY uint256_g EMBEDDED_FIELD_R; 
extern DEVICE_VAR CONST_MEMORY uint256_g EMBEDDED_FIELD_R_inv; 
extern DEVICE_VAR CONST_MEMORY uint32_t EMBEDDED_FIELD_N;

// Base field
extern DEVICE_VAR CONST_MEMORY uint256_g BASE_FIELD_P;
extern DEVICE_VAR CONST_MEMORY uint32_t BASE_FIELD_N;
extern DEVICE_VAR CONST_MEMORY uint256_g BASE_FIELD_R; 
extern DEVICE_VAR CONST_MEMORY uint256_g BASE_FIELD_R2;
extern DEVICE_VAR CONST_MEMORY uint256_g BASE_FIELD_R3;
extern DEVICE_VAR CONST_MEMORY uint256_g BASE_FIELD_R4;
extern DEVICE_VAR CONST_MEMORY uint256_g BASE_FIELD_R8;

// MAGIC_POWER =(P+1)/4 is constant, so we are able to precompute it (it is needed for exponentiation in a finite field)
// Magic constant should be given in standard form (i.e. NON MONTGOMERY)

extern DEVICE_VAR CONST_MEMORY uint256_g MAGIC_CONSTANT;

// Elliptic curve params
// A = 0
extern DEVICE_VAR CONST_MEMORY uint256_g CURVE_A_COEFF;
// B = 3
extern DEVICE_VAR CONST_MEMORY uint256_g CURVE_B_COEFF;
// generator G = [1, 2, 1]
extern DEVICE_VAR CONST_MEMORY  ec_point CURVE_G;

// This fconstant is used in Kasinski algorithm: that is fast field inversion in Montgomety form
extern DEVICE_VAR CONST_MEMORY uint256_g BASE_FIELD_R_SQUARED;

// This is required for experiemental version of mont mul
extern DEVICE_VAR CONST_MEMORY uint256_g BASE_FIELD_N_LARGE;

// This are used for FFT
extern DEVICE_FUNC size_t ROOTS_OF_UNTY_ARR_LEN;
extern DEVICE_FUNC CONST_MEMORY uint256_g EMBEDDED_FIELD_ROOTS_OF_UNITY[]; 

extern DEVICE_FUNC size_t MULT_GEN_ARR_LEN;
extern DEVICE_FUNC CONST_MEMORY uint256_g EMBEDDED_FIELD_MULT_GEN_ARR[];

extern DEVICE_FUNC size_t MULT_GEN_INV_ARR_LEN;
extern DEVICE_FUNC CONST_MEMORY uint256_g EMBEDDED_FIELD_MULT_GEN_INV_ARR[];

// A bunch of helpful structs
// We are not able to compile with C++ 17 standard :(
struct none_t{};
extern DEVICE_VAR CONST_MEMORY none_t NONE_OPT;

template<typename T>
class optional
{
private:
    bool flag_;
    T val_;

    static_assert(std::is_default_constructible<T>::value, "Inner type of optional should be constructible!");
public:
    DEVICE_FUNC optional(const T& val): flag_(true), val_(val) {}
    DEVICE_FUNC optional(const none_t& none): flag_(false) {}
    DEVICE_FUNC optional(): flag_(false) {}

    DEVICE_FUNC operator bool() const
    {
        return flag_;
    }

    DEVICE_FUNC const T& get_val() const
    {
        assert(flag_);
        return val_;
    } 
};


// Device specific functions
DEVICE_FUNC uint128_with_carry_g add_uint128_with_carry_asm(const uint128_g&, const uint128_g&);
DEVICE_FUNC uint128_g sub_uint128_asm(const uint128_g&, const uint128_g&);
DEVICE_FUNC uint256_g add_uint256_naive(const uint256_g&, const uint256_g&);
DEVICE_FUNC uint256_g add_uint256_asm(const uint256_g&, const uint256_g&);
DEVICE_FUNC uint256_g sub_uint256_naive(const uint256_g&, const uint256_g&);
DEVICE_FUNC uint256_g sub_uint256_asm(const uint256_g&, const uint256_g&);
DEVICE_FUNC int cmp_uint256_naive(const uint256_g&, const uint256_g&);

DEVICE_FUNC void add_uint_uint256_asm(uint256_g&, uint32_t);
DEVICE_FUNC void sub_uint_uint256_asm(uint256_g&, uint32_t);

DEVICE_FUNC bool is_zero(const uint256_g&);
DEVICE_FUNC bool is_even(const uint256_g&);

DEVICE_FUNC uint256_g shift_right_asm(const uint256_g&, uint32_t);
DEVICE_FUNC uint256_g shift_left_asm(const uint256_g&, uint32_t);

#define CMP(a, b) cmp_uint256_naive(a, b)
#define ADD(a, b) add_uint256_asm(a, b)
#define SUB(a, b) sub_uint256_asm(a, b)
#define SHIFT_LEFT(a, b) shift_left_asm(a, b)
#define SHIFT_RIGHT(a, b) shift_right_asm(a, b)
#define ADD_UINT(a, b) add_uint_uint256_asm(a, b)
#define SUB_UINT(a, b) sub_uint_uint256_asm(a, b)

DEVICE_FUNC inline bool EQUAL(const uint256_g& lhs, const uint256_g& rhs)
{
    return CMP(lhs, rhs) == 0;
}

// Helper functions for naive multiplication 
DEVICE_FUNC inline uint32_t device_long_mul(uint32_t x, uint32_t y, uint32_t* high_ptr)
	{
		uint32_t high = __umulhi(x, y);
		*high_ptr = high;
		return x * y;
	}

DEVICE_FUNC inline uint32_t device_fused_add(uint32_t x, uint32_t y, uint32_t* high_ptr)
{
	uint32_t z = x + y;
	if (z < x)
		(*high_ptr)++;
    return z;
}

DEVICE_FUNC uint256_g mul_uint128_to_256_naive(const uint128_g&, const uint128_g&);
DEVICE_FUNC uint256_g mul_uint128_to_256_asm_ver1(const uint128_g&, const uint128_g&);

#if (__CUDA_ARCH__ >= 500)
DEVICE_FUNC uint256_g mul_uint128_to_256_asm_ver2(const uint128_g&, const uint128_g&);
#endif

#define MUL_SHORT(a, b) mul_uint128_to_256_asm_ver1(a, b)

DEVICE_FUNC uint512_g mul_uint256_to_512_naive(const uint256_g&, const uint256_g&);
DEVICE_FUNC uint512_g mul_uint256_to_512_asm(const uint256_g&, const uint256_g&);
DEVICE_FUNC uint512_g mul_uint256_to_512_asm_with_allocation(const uint256_g&, const uint256_g&);
DEVICE_FUNC uint512_g mul_uint256_to_512_asm_longregs(const uint256_g&, const uint256_g&);
DEVICE_FUNC uint512_g mul_uint256_to_512_Karatsuba(const uint256_g&, const uint256_g&);
DEVICE_FUNC uint512_g mul_uint256_to_512_asm_with_shuffle(const uint256_g&, const uint256_g&);

#define MUL(a, b) mul_uint256_to_512_asm_with_allocation(a, b)

DEVICE_FUNC uint512_g square_uint256_to_512_naive(const uint256_g&);
DEVICE_FUNC uint512_g square_uint256_to_512_asm(const uint256_g&);

DEVICE_FUNC uint256_g mont_mul_256_naive_SOS(const uint256_g&, const uint256_g&);
DEVICE_FUNC uint256_g mont_mul_256_naive_CIOS(const uint256_g&, const uint256_g&);
DEVICE_FUNC uint256_g mont_mul_256_asm_SOS(const uint256_g&, const uint256_g&);
DEVICE_FUNC uint256_g mont_mul_256_asm_CIOS(const uint256_g&, const uint256_g&);

#define MONT_SQUARE(a) mont_mul_256_asm_SOS(a, a)
#define MONT_MUL(a,b) mont_mul_256_asm_CIOS(a, b)

DEVICE_FUNC uint256_g FIELD_ADD(const uint256_g&, const uint256_g&);
DEVICE_FUNC uint256_g FIELD_SUB(const uint256_g&, const uint256_g&);
DEVICE_FUNC uint256_g FIELD_ADD_INV(const uint256_g&);
DEVICE_FUNC uint256_g FIELD_MUL_INV(const uint256_g&);

// Implementation of these routines doesn't depend on whether we consider prokective or jacobian coordinates
DEVICE_FUNC inline bool is_infinity(const ec_point& point)
{
    return is_zero(point.z);
}

DEVICE_FUNC inline ec_point point_at_infty()
{
    ec_point pt;
	
	// XXX: may be we should use asm and xor here?
	#pragma unroll
    for (int32_t i = 0 ; i < N_BASE; i++)
    {
        pt.x.n[i] = 0;
    }
    pt.y.n[0] = 1;
	#pragma unroll
    for (int32_t  i= 1 ; i < N_BASE; i++)
    {
        pt.y.n[i] = 0;
    }
	#pragma unroll
    for (int32_t i = 0 ; i < N_BASE; i++)
    {
        pt.z.n[i] = 0;
    }

	return pt;
}

DEVICE_FUNC inline ec_point INV(const ec_point& pt)
{
    return {pt.x, FIELD_ADD_INV(pt.y), pt.z};
}

DEVICE_FUNC ec_point ECC_DOUBLE_PROJ(const ec_point&);
DEVICE_FUNC bool IS_ON_CURVE_PROJ(const ec_point&);
DEVICE_FUNC bool EQUAL_PROJ(const ec_point&, const ec_point&);
DEVICE_FUNC ec_point ECC_ADD_PROJ(const ec_point&, const ec_point&);
DEVICE_FUNC ec_point ECC_SUB_PROJ(const ec_point&, const ec_point&);
DEVICE_FUNC ec_point ECC_ADD_MIXED_PROJ(const ec_point&, const affine_point&);

DEVICE_FUNC ec_point ECC_DOUBLE_JAC(const ec_point&);
DEVICE_FUNC bool IS_ON_CURVE_JAC(const ec_point&);
DEVICE_FUNC bool EQUAL_JAC(const ec_point&, const ec_point&);
DEVICE_FUNC ec_point ECC_ADD_JAC(const ec_point&, const ec_point&);
DEVICE_FUNC ec_point ECC_SUB_JAC(const ec_point&, const ec_point&);
DEVICE_FUNC ec_point ECC_ADD_MIXED_JAC(const ec_point&, const affine_point&);

DEVICE_FUNC ec_point ECC_double_and_add_exp_PROJ(const ec_point&, const uint256_g&);
DEVICE_FUNC ec_point ECC_ternary_expansion_exp_PROJ(const ec_point&, const uint256_g&);
DEVICE_FUNC ec_point ECC_double_and_add_exp_JAC(const ec_point&, const uint256_g&);
DEVICE_FUNC ec_point ECC_ternary_expansion_exp_JAC(const ec_point&, const uint256_g&);
DEVICE_FUNC ec_point ECC_double_and_add_affine_exp_PROJ(const affine_point&, const uint256_g&);
DEVICE_FUNC ec_point ECC_double_and_add_affine_exp_JAC(const affine_point&, const uint256_g&);
DEVICE_FUNC ec_point ECC_wNAF_exp_PROJ(const ec_point&, const uint256_g&);
DEVICE_FUNC ec_point ECC_wNAF_exp_JAC(const ec_point&, const uint256_g&);


#ifdef USE_PROJECTIVE_COORDINATES

#define ECC_ADD(a, b) ECC_ADD_PROJ(a, b)
#define ECC_SUB(a, b) ECC_SUB_PROJ(a, b)
#define ECC_DOUBLE(a) ECC_DOUBLE_PROJ(a)
#define ECC_EXP(p, d) ECC_double_and_add_affine_exp_PROJ(p, d)
#define IS_ON_CURVE(p) IS_ON_CURVE_PROJ(p)
#define ECC_MIXED_ADD(a, b) ECC_ADD_MIXED_PROJ(a, b)

#elif defined USE_JACOBIAN_COORDINATES

#define ECC_ADD(a, b) ECC_ADD_JAC(a, b)
#define ECC_SUB(a, b) ECC_SUB_JAC(a, b)
#define ECC_DOUBLE(a) ECC_DOUBLE_JAC(a)
#define ECC_EXP(p, d) ECC_double_and_add_affine_exp_JAC(p, d)
#define IS_ON_CURVE(p) IS_ON_CURVE_JAC(p)
#define ECC_MIXED_ADD(a, b) ECC_ADD_MIXED_JAC(a, b)

#else
#error The form of elliptic curve coordinates should be explicitely specified
#endif

// Random elements generators
DEVICE_FUNC void gen_random_elem(uint256_g&, curandState&);
DEVICE_FUNC void gen_random_elem(embedded_field&, curandState&);
DEVICE_FUNC void gen_random_elem(ec_point&, curandState&);
DEVICE_FUNC void gen_random_elem(affine_point&, curandState&);

template <typename T>
__global__ void gen_random_array_kernel(T* elems, size_t arr_len, curandState* state, int seed)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
    curand_init(seed + tid, 0, 0, &state[tid]);

    curandState localState = state[tid];

    while (tid < arr_len)
    {
        gen_random_elem(elems[tid], localState);
        tid += blockDim.x * gridDim.x;
    }
}

#endif

#endif