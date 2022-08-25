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

#include "kernel_utilities.cuh"
#define PI 3.1415926535897932384626433
#define TWOPI (2*PI)

namespace UltraCold
{
    namespace KernelUtilities
    {

        ///////////////////////////////////////////////////////
        // Pure device kernels
        ///////////////////////////////////////////////////////

        // A useful complex exponential function
        __device__ cuDoubleComplex complex_exponential(cuDoubleComplex input)
        {
            cuDoubleComplex res;
            double t = expf (input.x);
            sincos (input.y, &res.y, &res.x);
            res.x *= t;
            res.y *= t;
            return res;
        }

        ////////////////////////////////////////////////////
        // Global kernels
        ////////////////////////////////////////////////////

        /**
         *
         * @brief Calculate the square of a complex vector, storing the result in another complex vector
         *
         * */

        __global__ void square_vector(cuDoubleComplex* result,
                                      cuDoubleComplex* input,
                                      int size)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for (int i = index; i < size; i += stride)
            {
                result[i].x = input[i].x*input[i].x +
                              input[i].y*input[i].y;
                result[i].y = 0.0;
            }
        }
        /**
         *
         * @brief Calculate the square of a complex vector
         *
         * */

        __global__ void square_vector(double* result,
                                      cuDoubleComplex* input,
                                      int size)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for (int i = index; i < size; i += stride)
            {
                result[i] = input[i].x*input[i].x +
                            input[i].y*input[i].y;
            }
        }

        /**
         *
         * @brief Calculate the square of a real vector
         *
         * */

        __global__ void square_vector(double* result,
                                      double* input,
                                      int size)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for (int i = index; i < size; i += stride)
            {
                result[i] = input[i]*input[i];
            }
        }

        /**
         *
         * @brief Multiply a real vector times the square of a complex vector
         *
         * */

        __global__ void vector_average(double* result,
                                       double* input1,
                                       cuDoubleComplex* input2,
                                       int size)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for (int i = index; i < size; i += stride)
            {
                result[i] = input1[i]*(input2[i].x*input2[i].x+input2[i].y*input2[i].y);
            }
        }

        /**
         *
         * @brief Multiply two complex vectors. Overwrite the first one
         *
         * */

        __global__ void vector_multiplication(cuDoubleComplex* result,
                                              cuDoubleComplex* input,
                                              int size)
        {
            cuDoubleComplex temp;
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for (int i = index; i < size; i += stride)
            {
                temp = result[i];
                result[i].x = temp.x*input[i].x - temp.y*input[i].y;
                result[i].y = temp.y*input[i].x + temp.x*input[i].y;
            }
        }

        /**
         *
         * @brief Multiply a complex and a real vector. Overwrite the complex one
         *
         * */

        __global__ void vector_multiplication(cuDoubleComplex* result,
                                              double* input,
                                              int size)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for (int i = index; i < size; i += stride)
            {
                result[i].x = result[i].x*input[i];
                result[i].y = result[i].y*input[i];
            }
        }

        /**
          *
          * @brief Multiply two real vectors. Overwrite the first one
          *
          * */

        __global__ void vector_multiplication(double* result,
                                              double* input,
                                              int size)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for (int i = index; i < size; i += stride)
            {
                result[i] = result[i]*input[i];
            }
        }

        /**
         *
         * @brief Multiply two complex vectors in the case in which the result is a real one
         *
         * */

        __global__ void vector_multiplication(double* result,
                                              cuDoubleComplex* input1,
                                              cuDoubleComplex* input2,
                                              int size)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for (int i = index; i < size; i += stride)
            {
                result[i] = input1[i].x * input2[i].x;
            }
        }

        /**
         *
         * @brief Second step in the operator splitting method for gradient descent
         *
         * */

        __global__ void step_2_hpsi(cuDoubleComplex* hpsi,
                                    cuDoubleComplex* psi,
                                    double* Vext,
                                    double* scattering_length,
                                    int size)
        {

            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for (int i = index; i < size; i += stride)
            {
                hpsi[i].x = hpsi[i].x +
                            (Vext[i]
                            + 4*PI*scattering_length[0]*(psi[i].x*psi[i].x+psi[i].y*psi[i].y)
                            ) *
                            psi[i].x;
                hpsi[i].y = hpsi[i].y +
                            (Vext[i]
                             + 4*PI*scattering_length[0]*(psi[i].x*psi[i].x+psi[i].y*psi[i].y)
                            ) *
                            psi[i].y;
            }
        }

        /**
         *
         * @brief Second step in the operator splitting method for gradient descent for dipolars
         *
         * */

        __global__ void step_2_dipolar_hpsi(cuDoubleComplex* hpsi,
                                            cuDoubleComplex* psi,
                                            double* Vext,
                                            cuDoubleComplex* Phi_dd,
                                            double* scattering_length,
                                            double* gamma_epsilon_dd,
                                            int size)
        {

            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            double aux=0.0;
            for (int i = index; i < size; i += stride)
            {
                aux= cuCabs(psi[i]);
                hpsi[i].x = hpsi[i].x +
                            (Vext[i]
                            + 4*PI*scattering_length[0]*pow(aux,2)
                            + Phi_dd[i].x
                            + gamma_epsilon_dd[0]*pow(aux,3)
                            ) *
                            psi[i].x;
                hpsi[i].y = hpsi[i].y +
                            (Vext[i]
                             + 4*PI*scattering_length[0]*pow(aux,2)
                             + Phi_dd[i].x
                             + gamma_epsilon_dd[0]*pow(aux,3)
                            ) *
                            psi[i].y;
            }
        }

        /**
         *
         * @brief Rescale a vector for a given input scalar
         *
         * */

        __global__ void rescale(cuDoubleComplex* result,
                                double input,
                                int size)
        {

            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for (int i = index; i < size; i += stride)
            {
                result[i].x = result[i].x * input;
                result[i].y = result[i].y * input;
            }

        }

        /**
         *
         * @brief Rescale a vector for a given input scalar, storing the result in another vector
         *
         * */

        __global__ void rescale(cuDoubleComplex* result,
                                cuDoubleComplex* input1,
                                double input2,
                                int size)
        {

            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for (int i = index; i < size; i += stride)
            {
                result[i].x = input1[i].x * input2;
                result[i].y = input1[i].y * input2;
            }

        }

        /**
         * @brief Gradient descent plus heavy-ball step
         *
         * */

        __global__ void gradient_descent_step(cuDoubleComplex* psi,
                                              cuDoubleComplex* hpsi,
                                              cuDoubleComplex* psi_new,
                                              cuDoubleComplex* psi_old,
                                              double* alpha,
                                              double* beta,
                                              int size)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for (int i = index; i < size; i += stride)
            {
                psi_new[i].x = (1.0 + beta[0])*psi[i].x - alpha[0]*hpsi[i].x - beta[0]*psi_old[i].x;
                psi_old[i].x = psi[i].x;
                psi_new[i].y = (1.0 + beta[0])*psi[i].y - alpha[0]*hpsi[i].y - beta[0]*psi_old[i].y;
                psi_old[i].y = psi[i].y;
            }
        }

        /**
         * @brief Solve step-1 operator splitting
         *
         * */

        __global__ void step_1_operator_splitting(cuDoubleComplex* psi,
                                                  double* Vext,
                                                  double* time_step,
                                                  double* scattering_length,
                                                  int size)
        {
            cuDoubleComplex aux;
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for (int i = index; i < size; i += stride)
            {
                aux.x = 0.0;
                aux.y = - time_step[0] * (Vext[i] + 4*PI*scattering_length[0]*(psi[i].x*psi[i].x+psi[i].y*psi[i].y) );
                psi[i] = cuCmul(psi[i],complex_exponential(aux));
            }
        }

        /**
         * @brief Solve step-1 operator splitting for dipolars
         *
         * */

        __global__ void step_1_operator_splitting_dipolars(cuDoubleComplex* psi,
                                                           double* Vext,
                                                           cuDoubleComplex* Phi_dd,
                                                           double* time_step,
                                                           double* scattering_length,
                                                           double* gamma_epsilon_dd,
                                                           int size)
        {
            cuDoubleComplex aux;
            double aux2;
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for (int i = index; i < size; i += stride)
            {
                aux2 = cuCabs(psi[i]);
                aux.x = 0.0;
                aux.y = - time_step[0] * (Vext[i]
                                          + 4*PI*scattering_length[0]*pow(aux2,2)
                                          + Phi_dd[i].x
                                          + gamma_epsilon_dd[0]*pow(aux2,3)
                                          );
                psi[i] = cuCmul(psi[i],complex_exponential(aux));
            }
        }

        /**
         *
         * @brief A useful help for step-2 of operator splitting
         *
         * */

        __global__ void aux_step_2_operator_splitting(cuDoubleComplex* psitilde,
                                                      double* kmod2,
                                                      double* time_step,
                                                      int size)
        {
            cuDoubleComplex aux;
	    aux.x = 0.0;
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;
            for (int i = index; i < size; i += stride)
            {
                aux.y = - 0.5 * time_step[0] * pow(TWOPI,2) * kmod2[i];
                psitilde[i] = cuCmul(psitilde[i],complex_exponential(aux));
            }
        }
    }
}
