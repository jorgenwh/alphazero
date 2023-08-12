#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(alphazero_C, m) {
  m.doc() "alphazero_C module";
}
