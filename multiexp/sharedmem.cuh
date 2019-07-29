#pragma once

template <typename EC>
struct SharedMemory {
    __device__ EC *getPointer() {
        extern __device__ void error(void);
        error();
        return NULL;
    }
};

template <>
struct SharedMemory <ECp_MNT4> {
    __device__ ECp_MNT4 *getPointer() {
        extern __shared__ ECp_MNT4 s_ecp_mnt4[];
        return s_ecp_mnt4;
    }
};

template <>
struct SharedMemory <ECp2_MNT4> {
    __device__ ECp2_MNT4 *getPointer() {
        extern __shared__ ECp2_MNT4 s_ecp2_mnt4[];
        return s_ecp2_mnt4;
    }
};

template <>
struct SharedMemory <ECp_MNT6> {
    __device__ ECp_MNT6 *getPointer() {
        extern __shared__ ECp_MNT6 s_ecp_mnt6[];
        return s_ecp_mnt6;
    }
};

template <>
struct SharedMemory <ECp3_MNT6> {
    __device__ ECp3_MNT6 *getPointer() {
        extern __shared__ ECp3_MNT6 s_ecp3_mnt6[];
        return s_ecp3_mnt6;
    }
};
