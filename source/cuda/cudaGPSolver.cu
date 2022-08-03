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

#include "cudaGPSolver.cuh"
#include "mesh_fourier_space.hpp"

#define PI 3.1415926535897932384626433
#define TWOPI (2*PI)

namespace UltraCold
{
    namespace cudaSolvers
    {
        /**
         * @brief Constructor for 1d problems
         *
         *
         *
         * */


        GPSolver::GPSolver(Vector<double> &x, Vector<std::complex<double>> &psi_0, Vector<double> &Vext,
                           double scattering_length)
        {

            // Get necessary copies of input data

            x_d.reinit(x);
            Vext_d.reinit(Vext);
            psi_d.reinit(psi_0);
            this->scattering_length = scattering_length;
            // Check the dimensions of the Vectors provided are consistent

            nx = x.extent(0);
            ny = 0;
            nz = 0;

            if(psi_0.extent(0) != nx || Vext.extent(0) != nx)
            {
                std::cout
                        << "\n\n"
                        << "**************************************************************************\n"
                        << "Error found in the constructor of a (cuda) GPSolver for a Gross-Pitaevskii\n"
                        << "equation in one space dimension. The dimensions of the Vectors provided as\n"
                        << "input are not consistent.\n"
                        << "Terminating the program now...\n"
                        << "**************************************************************************\n"
                        << "\n\n"
                        <<
                        std::endl;
                exit(1);
            }

            // Initialize the mesh in Fourier space

            Vector<double> kx(nx);
            Vector<double> kmod2(nx);

            create_mesh_in_Fourier_space(x,kx);
            for (size_t i = 0; i < nx; ++i)
                kmod2(i) = std::pow(kx(i),2);

            kx_d.reinit(kx);
            kmod2_d.reinit(kmod2);

            psitilde_d.reinit(nx);
            hpsi_d.reinit(nx);

            // Initialize space steps

            dx = x(1)-x(0);
            dv = dx;

            // calculate initial norm



        }

        /**
         * Definition of global kernels
         *
         * */

        __global__ void calculate_norm(double norm)
        {
            thrust::device_ptr<std::complex<double>> my_ptr;
        };
    }
}