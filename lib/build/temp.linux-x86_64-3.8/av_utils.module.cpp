#include <pybind11/pybind11.h>
void pybind_output_fun_utils_cpp(pybind11::module&);
PYBIND11_MODULE(av_utils, m) {
m.doc() = "TODO: Dodumentation";
pybind_output_fun_utils_cpp(m);
#ifdef VERSION_INFO
m.attr("__version__") = VERSION_INFO;
m.attr("__version__") = "dev";
#endif
}
