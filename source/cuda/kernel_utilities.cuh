/*--------------------------------------------------------------------------------
*
*    This file is part of the UltraCold project.
*
*    UltraCold is free software: you can redistribute it and/or modify
*    it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    any later version.
*    UltraCold is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU General Public License for more details.
*    You should have received a copy of the GNU General Public License
*    along with UltraCold.  If not, see <https://www.gnu.org/licenses/>.
*
*--------------------------------------------------------------------------------*/

#ifndef ULTRACOLD_CUDA_KERNELS
#define ULTRACOLD_CUDA_KERNELS

#include "cuComplex.h"
#include "cufft.h"

namespace UltraCold
{
    namespace KernelUtilities
    {
        // Device kernels

        __device__ cuDoubleComplex complex_exponential(cuDoubleComplex input);

        // Global kernels

        // General utilities

        __global__ void square_vector( cuDoubleComplex* result,
                                       cuDoubleComplex* input,
                                       int size);
        __global__ void square_vector( double* result,
                                       cuDoubleComplex* input,
                                       int size);
        __global__ void square_vector(double* result,
                                      double* input,
                                      int size);
        __global__ void vector_average(double* result,
                                       double* input1,
                                       cuDoubleComplex* input2,
                                       int size);
        __global__ void vector_multiplication(cuDoubleComplex* result,
                                              cuDoubleComplex* input,
                                              int size);
        __global__ void vector_multiplication(cuDoubleComplex* result,
                                              double* input,
                                              int size);
        __global__ void vector_multiplication(double* result,
                                              double* input,
                                              int size);
        __global__ void vector_multiplication(double* result,
                                              cuDoubleComplex* input1,
                                              cuDoubleComplex* input2,
                                              int size);
        __global__ void vector_multiplication(double* result,
                                              cuDoubleComplex* v1,
                                              double* v2,
                                              int size);
        __global__ void rescale(cuDoubleComplex* result,
                                double input,
                                int size);
        __global__ void rescale(cuDoubleComplex* result,
                                cuDoubleComplex* input1,
                                double input2,
                                int size);

        // More specific ones

        __global__ void step_2_hpsi(cuDoubleComplex* hpsi,
                                    cuDoubleComplex* psi,
                                    double* Vext,
                                    double* scattering_length,
                                    int size);
        __global__ void step_2_dipolar_hpsi(cuDoubleComplex* hpsi,
                                            cuDoubleComplex* psi,
                                            double* Vext,
                                            cuDoubleComplex* Phi_dd,
                                            double* scattering_length,
                                            double* gamma_epsilon_dd,
                                            int size);
        __global__ void gradient_descent_step(cuDoubleComplex* psi,
                                              cuDoubleComplex* hpsi,
                                              cuDoubleComplex* psi_new,
                                              cuDoubleComplex* psi_old,
                                              double* alpha,
                                              double* beta,
                                              int size);
        __global__ void step_1_operator_splitting(cuDoubleComplex* psi,
                                                  double* external_potential,
                                                  double* time_step,
                                                  double* scattering_length,
                                                  int size);
        __global__ void aux_step_2_operator_splitting(cuDoubleComplex* psitilde,
                                                      double* kmod2,
                                                      double* time_step,
                                                      int size);
        __global__ void step_1_operator_splitting_dipolars(cuDoubleComplex* psi,
                                                           double* Vext,
                                                           cuDoubleComplex* Phi_dd,
                                                           double* time_step,
                                                           double* scattering_length,
                                                           double* gamma_epsilon_dd,
                                                           int size);
    }
}

#endif // ULTRACOLD_CUDA_KERNELS