/*! @file
 * @brief cstone DeviceVector with the additional instantiation for the 98-double multipole type of the test
 */

#include "cstone/cuda/device_vector.cu"

namespace cstone
{
template class DeviceVector<util::array<double, 98>>;
}
