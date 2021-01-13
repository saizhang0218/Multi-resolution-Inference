#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <vector>

#define MAX_GROUP_SIZE 32
#define MAX_TERMS 64
#define MAX_VALUES 64

namespace {
template <typename scalar_t>
__device__ void linear_encode(const scalar_t input,
                              int32_t *__restrict__ terms,
                              int32_t *num_terms,
                              const int32_t bitwidth,
                              const float sf) {
  int32_t b1;
  float maxv = pow(2, bitwidth) - 1;
  int32_t q_val = fminf(int32_t(abs(input) / sf + 0.5), maxv);
  int32_t sign = input < 0 ? -1 : 1;
  *num_terms = 0;
  for (int i = 0; i < MAX_TERMS; i++) {
    terms[i] = 0;
  }

  for (int i = MAX_TERMS - 1; i >= 0; i--) {
    b1 = (q_val >> i) & 1;

    if (b1 == 1) {
      terms[*num_terms] = sign * (1 << i);
      (*num_terms)++;
    }
  }
}

template <typename scalar_t>
__device__ void hese_encode(const scalar_t input,
                            int32_t *__restrict__ terms,
                            int32_t *num_terms,
                            const int32_t bitwidth,
                            const float sf) {
  int32_t b0, b1, b2;
  float maxv = pow(2, bitwidth) - 1;
  int32_t q_val = fminf(int32_t(abs(input) / sf + 0.5), maxv);
  int32_t sign = input < 0 ? -1 : 1;
  *num_terms = 0;
  for (int i = 0; i < MAX_TERMS; i++) {
    terms[i] = 0;
  }

  for (int i = MAX_TERMS - 1; i >= 0; i--) {
    b0 = i == 0 ? 0 : (q_val >> (i - 1)) & 1;
    b1 = (q_val >> i) & 1;
    b2 = i == MAX_TERMS - 1 ? 0 : (q_val >> (i + 1)) & 1;

    if (b2 == 0 && b1 == 0 && b0 == 0) {
      continue;
    } else if (b2 == 0 && b1 == 0 && b0 == 1) {
      continue;
    } else if (b2 == 0 && b1 == 1 && b0 == 0) {
      terms[*num_terms] = sign * (1 << i);
      (*num_terms)++;
      i--;
    } else if (b2 == 0 && b1 == 1 && b0 == 1) {
      terms[*num_terms] =  sign * (1 << (i+1));
      (*num_terms)++;
    } else if (b2 == 1 && b1 == 0 && b0 == 0) {
      continue;
    } else if (b2 == 1 && b1 == 0 && b0 == 1) {
      continue;
    } else if (b2 == 1 && b1 == 1 && b0 == 0) {
      terms[*num_terms] = (-sign) * (1 << i);
      (*num_terms)++;
    } else if (b2 == 1 && b1 == 1 && b0 == 1) {
      continue;
    }
  }
}

template <typename scalar_t>
__device__ void optim_encode(const scalar_t input,
                             int32_t *__restrict__ terms,
                             int32_t *num_terms,
                             const int32_t bitwidth,
                             const float sf) {
  int32_t b0, b1;
  float maxv = pow(2, bitwidth) - 1;
  int32_t q_val = fminf(int32_t(abs(input) / sf + 0.5), maxv);
  int32_t sign = input < 0 ? -1 : 1;
  int32_t mode = 0;
  *num_terms = 0;
  for (int i = 0; i < MAX_TERMS; i++) {
    terms[i] = 0;
  }

  int32_t loc_terms[MAX_TERMS];

  for (int i = 0; i < MAX_TERMS; i++) {
    b0 = (q_val >> i) & 1;
    b1 = i == MAX_TERMS - 1 ? 0 : (q_val >> (i + 1)) & 1;

    if (mode == 0 && b1 == 0 && b0 == 0) {
      ;
    } else if (mode == 0 && b1 == 0 && b0 == 1) {
      loc_terms[*num_terms] = sign * (1 << i);
      (*num_terms)++;
    } else if (mode == 0 && b1 == 1 && b0 == 0) {
      ;
    } else if (mode == 0 && b1 == 1 && b0 == 1) {
      loc_terms[*num_terms] = (-1*sign) * (1 << i);
      (*num_terms)++;
      mode = 1;
    } else if (mode == 1 && b1 == 0 && b0 == 0) {
      loc_terms[*num_terms] = sign * (1 << i);
      (*num_terms)++;
      mode = 0;
    } else if (mode == 1 && b1 == 0 && b0 == 1) {
      ;
    } else if (mode == 1 && b1 == 1 && b0 == 0) {
      loc_terms[*num_terms] = (-1*sign) * (1 << i);
      (*num_terms)++;
    } else if (mode == 1 && b1 == 1 && b0 == 1) {
      ;
    }
  }

  //reverse order
  for (int i = 0; i < *num_terms; i++) {
    terms[i] = loc_terms[*num_terms - i - 1];
  }
}

template <typename scalar_t>
__global__ void tr_cuda_kernel(const scalar_t *__restrict__ input,
                               scalar_t *__restrict__ output,
                               const float sf,
                               const int32_t quantization_type,
                               const int32_t bitwidth,
                               const int32_t group_size,
                               const int32_t num_keep_terms,
                               const int32_t B,
                               const int32_t C,
                               const int32_t W,
                               const int32_t H) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t CWH = C * W * H;
  const int32_t WH = W * H;
  const int32_t b = idx / CWH;
  const int32_t c = (idx - b * CWH) / WH;
  const int32_t w = (idx - b * CWH - c * WH) / H;
  const int32_t h = idx - b * CWH - c * WH - w * H;
  const int32_t base_offset = b * CWH + w * W + h;
  const int32_t max_size = int(ceilf(C / float(group_size)));
  int32_t gidx;

  if (b < B && c < max_size && w < W && h < H) {
    int32_t term_idx[MAX_GROUP_SIZE];
    int32_t num_terms[MAX_GROUP_SIZE];
    int32_t terms[MAX_GROUP_SIZE * MAX_TERMS];
    for (int i = 0; i < group_size; ++i) {
      gidx = (c * group_size + i) * WH + base_offset;
      output[gidx] = 0;
      term_idx[i] = 0;
      if (quantization_type == 0) {
        linear_encode(input[gidx], &terms[i * MAX_TERMS], &num_terms[i], bitwidth, sf);
      } else if (quantization_type == 1) {
        hese_encode(input[gidx], &terms[i * MAX_TERMS], &num_terms[i], bitwidth, sf);
      } else if (quantization_type == 2) {
        optim_encode(input[gidx], &terms[i * MAX_TERMS], &num_terms[i], bitwidth, sf);
      }
    }

    for (int i = 0; i < num_keep_terms; ++i) {
      int32_t max_idx = 0;
      int32_t max_val = 0;
      // loop through groups and add max term
      for (int j = 0; j < group_size; ++j) {
        // find maximum term (of sorted choices)
        int32_t term = terms[j * MAX_TERMS + term_idx[j]];
        if (abs(term) > abs(max_val)) {
          max_val = term;
          max_idx = j;
        }
      }

      // no more useful terms left -- exit
      if (max_val == 0) {
        break;
      }

      // add the max term to correct output
      gidx = (c * group_size + max_idx) * WH + base_offset;
      output[gidx] += max_val;

      // increment pointer to max term element
      term_idx[max_idx]++;
    }

    // multiply each entry by scaling factor
    for (int i = 0; i < group_size; ++i) {
      gidx = (c * group_size + i) * WH + base_offset;
      int32_t sign = input[gidx] < 0 ? -1 : 1;
      output[gidx] *= sf;
    }
  }
}

template <typename scalar_t>
__global__ void term_trunc_cuda_kernel(const scalar_t *__restrict__ input,
                                       scalar_t *__restrict__ output,
                                       const float sf,
                                       const int32_t quantization_type,
                                       const int32_t bitwidth,
                                       const int32_t num_keep_terms,
                                       const int32_t size) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    int32_t num_terms[1];
    int32_t terms[MAX_TERMS];
    output[idx] = 0;

    if (quantization_type == 0) {
      linear_encode(input[idx], &terms[0], &num_terms[0], bitwidth, sf);
    } else if (quantization_type == 1) {
      hese_encode(input[idx], &terms[0], &num_terms[0], bitwidth, sf);
    } else if (quantization_type == 2) {
      optim_encode(input[idx], &terms[0], &num_terms[0], bitwidth, sf);
    }

    for (int i = 0; i < num_keep_terms; ++i) {
      if (i == num_terms[0]) break;
      output[idx] += terms[i];
    }

    output[idx] *= sf;
  }
}

} // namespace

at::Tensor tr_cuda(const at::Tensor input,
                   const float sf,
                   const int32_t quantization_type,
                   const int32_t bitwidth,
                   const int32_t group_size,
                   const int32_t num_keep_terms) {
  const auto ndim = input.ndimension();
  const auto B = input.size(0);
  const auto C = input.size(1);
  auto W = 1;
  auto H = 1;
  if (ndim == 4) {
    W = input.size(2);
    H = input.size(3);
  }
  const auto size = B * C * W * H;
  const int threads = 128;
  const int blocks = (size + threads - 1) / threads;
  auto output = at::zeros_like(input);
  AT_DISPATCH_FLOATING_TYPES(input.type(), "tr_cuda", ([&] {
                               tr_cuda_kernel<scalar_t><<<blocks, threads>>>(
                                   input.data<scalar_t>(),
                                   output.data<scalar_t>(),
                                   sf,
                                   quantization_type,
                                   bitwidth,
                                   group_size,
                                   num_keep_terms,
                                   B,
                                   C,
                                   W,
                                   H);
                             }));
  return output;
}

at::Tensor term_trunc_cuda(const at::Tensor input,
                           const float sf,
                           const int32_t quantization_type,
                           const int32_t bitwidth,
                           const int32_t num_keep_terms) {
  const auto ndim = input.ndimension();
  const auto B = input.size(0);
  const auto C = input.size(1);
  auto W = 1;
  auto H = 1;
  if (ndim == 4) {
    W = input.size(2);
    H = input.size(3);
  }
  const auto size = B * C * W * H;
  const int threads = 128;
  const int blocks = (size + threads - 1) / threads;
  auto output = at::zeros_like(input);
  AT_DISPATCH_FLOATING_TYPES(input.type(), "term_trunc_cuda", ([&] {
                               term_trunc_cuda_kernel<scalar_t><<<blocks, threads>>>(
                                   input.data<scalar_t>(),
                                   output.data<scalar_t>(),
                                   sf,
                                   quantization_type,
                                   bitwidth,
                                   num_keep_terms,
                                   size);
                             }));
  return output;
}