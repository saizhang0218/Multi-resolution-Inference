#include <torch/torch.h>

#include <vector>

// CUDA forward declarations
at::Tensor tr_cuda(const at::Tensor input, const float sf, const int32_t quantization_type,
                   const int32_t bitwidth, const int32_t group_size,
                   const int32_t num_keep_terms);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

at::Tensor tr(const at::Tensor input, const float sf, const int32_t quantization_type,
              const int32_t bitwidth, const int32_t group_size, const int32_t num_keep_terms) {
  CHECK_INPUT(input);
  return tr_cuda(input, sf, quantization_type, bitwidth, group_size, num_keep_terms);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tr", &tr, "Term Revealing (TR) (CUDA)");
}
