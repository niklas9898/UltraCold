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

#ifndef ULTRACOLD_CUDA_GP_SOLVER
#define ULTRACOLD_CUDA_GP_SOLVER

#include "cudaDeviceVector.cuh"
#include <vector>
#include <cub/cub.cuh>

namespace UltraCold
{

    /**
     * @brief This is a simple solver for the GP equation using CUDA
     * @author Santo Maria Roccuzzo (santom.roccuzzo@gmail.com)
     *
     * */

    namespace cudaSolvers
    {

        /**
         * Bunch of global kernel declarations
         * */

        __global__ void calculate_norm(double norm);

        class GPSolver
        {
            public:

                // Constructors

                GPSolver(Vector<double>&               x,
                         Vector<std::complex<double>>& psi_0,
                         Vector<double>&               Vext,
                         double                        scattering_length);  // 1D problems

                GPSolver(Vector<double>&               x,
                         Vector<double>&               y,
                         Vector<std::complex<double>>& psi_0,
                         Vector<double>&               Vext,
                         double                        scattering_length);  // 2D problems

                GPSolver(Vector<double>&               x,
                         Vector<double>&               y,
                         Vector<double>&               z,
                         Vector<std::complex<double>>& psi_0,
                         Vector<double>&               Vext,
                         double                        scattering_length);  // 3D problems

                // Destructor

                ~GPSolver()=default;

                // Calculate a ground-state solution

                std::tuple<Vector<std::complex<double>>,double> run_gradient_descent(int    max_num_iter,
                                                                                     double tolerance,
                                                                                     double alpha,
                                                                                     double beta,
                                                                                     std::ostream& output_stream);

                // Calculate current norm

                void calculate_norm_h(double norm)
                {
                    calculate_norm<<<1,1>>> (norm);
                };


            private:

                // Vector data members, needed for the calculations
                // Vectors living in device memory have the _d postfix

                cudaDeviceVector<std::complex<double>> psi_d;
                cudaDeviceVector<double>    Vext_d;
                cudaDeviceVector<double>    x_d;
                cudaDeviceVector<double>    y_d;
                cudaDeviceVector<double>    z_d;
                cudaDeviceVector<double>    kx_d;
                cudaDeviceVector<double>    ky_d;
                cudaDeviceVector<double>    kz_d;
                cudaDeviceVector<double>    kmod2_d;               // This contains the squared moduli of the k-vectors
                cudaDeviceVector<std::complex<double>> psitilde_d; // This contains the Fourier transform of psi
                cudaDeviceVector<std::complex<double>> hpsi_d;     // This contains the result of \hat{H}\psi

                // Number of points along each space dimension

                int nx,ny,nz;

                // Other mesh parameter

                double dx = 1.0;
                double dy = 1.0;
                double dz = 1.0;
                double dv = 1.0;

                // Other useful parameters

                double chemical_potential;
                double scattering_length;
                double residual;
                double initial_norm;
                double norm;
                double time_step;
                std::complex<double> ci={0.0,1.0};
                int last_iteration_number;

                // Eigenstates of the harmonic oscillator, useful for TWA

                std::vector<Vector<double>> eigenstates_harmonic_oscillator;
        };
    }

}


#endif