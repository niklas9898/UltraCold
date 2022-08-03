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

#include "cudaDipolarGPSolver.cuh"
#include "mesh_fourier_space.hpp"
#include "DataOut.hpp"
#include "cub/cub.cuh"
#include "cufft.h"
#include "kernel_utilities.cuh"

#define PI 3.1415926535897932384626433
#define TWOPI (2*PI)

namespace UltraCold
{
    namespace cudaSolvers
    {

        /////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief Constructor for 2d problems
         * */
        /////////////////////////////////////////////////////////////////////////////////////

        DipolarGPSolver::DipolarGPSolver(Vector<double> &x,
                                         Vector<double> &y,
                                         Vector<std::complex<double>> &psi_0,
                                         Vector<double> &Vext,
                                         double scattering_length,
                                         double dipolar_length)
        {

            // Check that the order of the Vectors provided is correctly 1
            if(psi_0.order() != 2 || Vext.order() != 2)
            {
                std::cout
                        << "\n\n"
                        << "**************************************************************************\n"
                        << "Error found in the constructor of a (cuda) GPSolver for a Gross-Pitaevskii\n"
                        << "equation in two space dimension. The orders of the Vectors provided as\n"
                        << "input are not consistent. In particular, initial wave function and external\n"
                        << "potential provided are not 2-dimensional.\n"
                        << "Terminating the program now...\n"
                        << "**************************************************************************\n"
                        << "\n\n"
                        <<
                        std::endl;
                exit(1);
            }
            problem_is_2d=true;

            // Check the dimensions of the Vectors provided are consistent
            nx = x.extent(0);
            ny = y.extent(0);
            nz = 0;
            npoints=nx*ny;
            if(psi_0.extent(0) != nx || Vext.extent(0) != nx ||
               psi_0.extent(1) != ny || Vext.extent(1) != ny)
            {
                std::cout
                        << "\n\n"
                        << "**************************************************************************\n"
                        << "Error found in the constructor of a (cuda) GPSolver for a Gross-Pitaevskii\n"
                        << "equation in two space dimension. The dimensions of the Vectors provided as\n"
                        << "input are not consistent.\n"
                        << "Terminating the program now...\n"
                        << "**************************************************************************\n"
                        << "\n\n"
                        <<
                        std::endl;
                exit(1);
            }

            // Initialize the thread grid, i.e. choose the number of cuda threads per block and the number of blocks.
            blockSize = 512;
            gridSize = (npoints + blockSize - 1) / blockSize;

            // Allocate memory for all device arrays
            cudaMalloc(&external_potential_d,npoints*sizeof(double));
            cudaMalloc(&kmod2_d,             npoints*sizeof(double));
            cudaMalloc(&density_d,           npoints*sizeof(double));
            cudaMalloc(&wave_function_d,     npoints*sizeof(cuDoubleComplex));
            cudaMalloc(&hpsi_d,              npoints*sizeof(cuDoubleComplex));
            cudaMalloc(&ft_wave_function_d,  npoints*sizeof(cuDoubleComplex));

            // Allocate space for device and managed scalars
            cudaMalloc(&scattering_length_d,sizeof(double));
            cudaMallocManaged(&norm_d,              sizeof(double));
            cudaMallocManaged(&initial_norm_d,      sizeof(double));
            cudaMallocManaged(&chemical_potential_d,sizeof(double));

            // Get the first necessary copies of input data from host to device
            cudaMemcpy(external_potential_d,Vext.data(),       npoints*sizeof(double),         cudaMemcpyHostToDevice);
            cudaMemcpy(wave_function_d,     psi_0.data(),      npoints*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
            cudaMemcpy(scattering_length_d, &scattering_length,1      *sizeof(double),         cudaMemcpyHostToDevice);

            // Initialize the mesh in Fourier space, and copy it to the device
            Vector<double> kx(nx);
            Vector<double> ky(ny);
            Vector<double> kmod2(nx,ny);
            create_mesh_in_Fourier_space(x,y,kx,ky);
            for (size_t i = 0; i < nx; ++i)
                for (size_t j = 0; j < ny; ++j)
                    kmod2(i,j) = std::pow(kx(i),2) +
                                 std::pow(ky(j),2);
            cudaMemcpy(kmod2_d,  kmod2.data(),npoints*sizeof(double),cudaMemcpyHostToDevice);

            // Initialize space steps
            dx = x(1)-x(0);
            dy = y(1)-y(0);
            dv = dx*dy;

            // Initialize the device reduce kernel
            cub::DeviceReduce::Sum(temporary_storage_d,size_temporary_storage,density_d,norm_d,npoints);
            cudaDeviceSynchronize();

            // Allocate temporary storage memory, required for reduction kernels
            cudaMalloc(&temporary_storage_d,size_temporary_storage);
            cudaDeviceSynchronize();

            // Calculate initial norm
            calculate_density(density_d,wave_function_d,npoints);
            cudaDeviceSynchronize();
            cub::DeviceReduce::Sum(temporary_storage_d,size_temporary_storage,density_d,norm_d,npoints);
            cudaDeviceSynchronize();
            norm_d[0]=norm_d[0]*dv;
            initial_norm_d[0]=norm_d[0];
            std::cout << "Initial norm: " << initial_norm_d[0] << std::endl;

            // Initialize the wave function to return as a result
            result_wave_function.reinit(nx,ny);

            // Initialize the host vectors containing the mesh axis. This can be useful in particular for data output
            x_axis.reinit(nx);
            y_axis.reinit(ny);
            x_axis=x;
            y_axis=y;
            kx_axis.reinit(nx);
            ky_axis.reinit(ny);
            kx_axis=kx;
            ky_axis=ky;
            cudaMalloc(&x_axis_d,nx*sizeof(double));
            cudaMalloc(&y_axis_d,ny*sizeof(double));
            cudaMemcpy(x_axis_d,x_axis.data(),nx*sizeof(double),cudaMemcpyHostToDevice);
            cudaMemcpy(y_axis_d,y_axis.data(),ny*sizeof(double),cudaMemcpyHostToDevice);
            cudaMalloc(&kx_axis_d,nx*sizeof(double));
            cudaMalloc(&ky_axis_d,ny*sizeof(double));
            cudaMemcpy(kx_axis_d,kx_axis.data(),nx*sizeof(double),cudaMemcpyHostToDevice);
            cudaMemcpy(ky_axis_d,ky_axis.data(),ny*sizeof(double),cudaMemcpyHostToDevice);
            r2mod.reinit(nx,ny);
            for(int i = 0; i < nx; ++i)
                for(int j = 0; j < ny; ++j)
                    r2mod(i,j) = std::pow(x(i),2)+std::pow(y(j),2);
            cudaMalloc(&r2mod_d,npoints*sizeof(double));
            cudaMemcpy(r2mod_d,r2mod.data(),npoints*sizeof(double),cudaMemcpyHostToDevice);
        }

        /////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief Constructor for 3d problems
         * */
        /////////////////////////////////////////////////////////////////////////////////////

        DipolarGPSolver::DipolarGPSolver(Vector<double> &x,
                                         Vector<double> &y,
                                         Vector<double> &z,
                                         Vector<std::complex<double>> &psi_0,
                                         Vector<double> &Vext,
                                         double scattering_length,
                                         double dipolar_length,
                                         double theta_mu,
                                         double phi_mu)
        {

            // Check that the order of the Vectors provided is correctly 1
            if(psi_0.order() != 3 || Vext.order() != 3)
            {
                std::cout
                        << "\n\n"
                        << "**************************************************************************\n"
                        << "Error found in the constructor of a (cuda) GPSolver for a Gross-Pitaevskii\n"
                        << "equation in three space dimension. The orders of the Vectors provided as\n"
                        << "input are not consistent. In particular, initial wave function and external\n"
                        << "potential provided are not 3-dimensional.\n"
                        << "Terminating the program now...\n"
                        << "**************************************************************************\n"
                        << "\n\n"
                        <<
                        std::endl;
                exit(1);
            }
            problem_is_3d=true;

            // Check the dimensions of the Vectors provided are consistent
            nx = x.extent(0);
            ny = y.extent(0);
            nz = z.extent(0);
            npoints=nx*ny*nz;
            if(psi_0.extent(0) != nx || Vext.extent(0) != nx ||
               psi_0.extent(1) != ny || Vext.extent(1) != ny ||
               psi_0.extent(2) != nz || Vext.extent(2) != nz)
            {
                std::cout
                        << "\n\n"
                        << "**************************************************************************\n"
                        << "Error found in the constructor of a (cuda) GPSolver for a Gross-Pitaevskii\n"
                        << "equation in three space dimension. The dimensions of the Vectors provided as\n"
                        << "input are not consistent.\n"
                        << "Terminating the program now...\n"
                        << "**************************************************************************\n"
                        << "\n\n"
                        <<
                        std::endl;
                exit(1);
            }

            // Initialize the thread grid, i.e. choose the number of cuda threads per block and the number of blocks.
            blockSize = 512;
            gridSize = (npoints + blockSize - 1) / blockSize;

            // Allocate memory for all device arrays
            cudaMalloc(&external_potential_d,npoints*sizeof(double));
            cudaMalloc(&kmod2_d,             npoints*sizeof(double));
            cudaMalloc(&density_d,           npoints*sizeof(double));
            cudaMalloc(&wave_function_d,     npoints*sizeof(cuDoubleComplex));
            cudaMalloc(&hpsi_d,              npoints*sizeof(cuDoubleComplex));
            cudaMalloc(&ft_wave_function_d,  npoints*sizeof(cuDoubleComplex));

            // Allocate space for device and managed scalars
            cudaMalloc(&scattering_length_d,sizeof(double));
            cudaMallocManaged(&norm_d,              sizeof(double));
            cudaMallocManaged(&initial_norm_d,      sizeof(double));
            cudaMallocManaged(&chemical_potential_d,sizeof(double));

            // Get the first necessary copies of input data from host to device
            cudaMemcpy(external_potential_d,Vext.data(),       npoints*sizeof(double),         cudaMemcpyHostToDevice);
            cudaMemcpy(wave_function_d,     psi_0.data(),      npoints*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
            cudaMemcpy(scattering_length_d, &scattering_length,1      *sizeof(double),         cudaMemcpyHostToDevice);

            // Initialize the mesh in Fourier space, and copy it to the device
            Vector<double> kx(nx);
            Vector<double> ky(ny);
            Vector<double> kz(nz);
            Vector<double> kmod2(nx,ny,nz);
            create_mesh_in_Fourier_space(x,y,z,kx,ky,kz);
            for (size_t i = 0; i < nx; ++i)
                for (size_t j = 0; j < ny; ++j)
                    for (size_t k = 0; k < nz; ++k)
                        kmod2(i,j,k) = std::pow(kx(i),2)+
                                       std::pow(ky(j),2)+
                                       std::pow(kz(k),2);
            cudaMemcpy(kmod2_d, kmod2.data(),npoints*sizeof(double),cudaMemcpyHostToDevice);

            // Initialize space steps
            dx = x(1)-x(0);
            dy = y(1)-y(0);
            dz = z(1)-z(0);
            dv = dx*dy*dz;

            // Initialize the device reduce kernel
            cub::DeviceReduce::Sum(temporary_storage_d,size_temporary_storage,density_d,norm_d,npoints);
            cudaDeviceSynchronize();

            // Allocate temporary storage memory, required for reduction kernels
            cudaMalloc(&temporary_storage_d,size_temporary_storage);
            cudaDeviceSynchronize();

            // Calculate initial norm
            calculate_density(density_d,wave_function_d,npoints);
            cudaDeviceSynchronize();
            cub::DeviceReduce::Sum(temporary_storage_d,size_temporary_storage,density_d,norm_d,npoints);
            cudaDeviceSynchronize();
            norm_d[0]=norm_d[0]*dv;
            initial_norm_d[0]=norm_d[0];
            std::cout << "Initial norm: " << initial_norm_d[0] << std::endl;

            // Initialize the wave function to return as a result
            result_wave_function.reinit(nx,ny,nz);

            // Initialize the host vectors containing the mesh axis. This can be useful in particular for data output
            x_axis.reinit(nx);
            y_axis.reinit(ny);
            z_axis.reinit(nz);
            x_axis=x;
            y_axis=y;
            z_axis=z;
            kx_axis.reinit(nx);
            ky_axis.reinit(ny);
            kz_axis.reinit(nz);
            kx_axis=kx;
            ky_axis=ky;
            kz_axis=kz;
            cudaMalloc(&x_axis_d,nx*sizeof(double));
            cudaMalloc(&y_axis_d,ny*sizeof(double));
            cudaMalloc(&z_axis_d,nz*sizeof(double));
            cudaMemcpy(x_axis_d,x_axis.data(),nx*sizeof(double),cudaMemcpyHostToDevice);
            cudaMemcpy(y_axis_d,y_axis.data(),ny*sizeof(double),cudaMemcpyHostToDevice);
            cudaMemcpy(z_axis_d,z_axis.data(),nz*sizeof(double),cudaMemcpyHostToDevice);
            cudaMalloc(&kx_axis_d,nx*sizeof(double));
            cudaMalloc(&ky_axis_d,ny*sizeof(double));
            cudaMalloc(&kz_axis_d,nz*sizeof(double));
            cudaMemcpy(kx_axis_d,kx_axis.data(),nx*sizeof(double),cudaMemcpyHostToDevice);
            cudaMemcpy(ky_axis_d,ky_axis.data(),ny*sizeof(double),cudaMemcpyHostToDevice);
            cudaMemcpy(kz_axis_d,kz_axis.data(),nz*sizeof(double),cudaMemcpyHostToDevice);
            r2mod.reinit(nx,ny,nz);
            for(int i = 0; i < nx; ++i)
                for(int j = 0; j < ny; ++j)
                    for(int k = 0; k < nz; ++k)
                        r2mod(i,j,k) = std::pow(x(i),2);
            cudaMalloc(&r2mod_d,npoints*sizeof(double));
            cudaMemcpy(r2mod_d,r2mod.data(),npoints*sizeof(double),cudaMemcpyHostToDevice);

            // Initialize the Fourier transform of the dipolar potential
            cudaMallocManaged(&epsilon_dd_d,sizeof(double));
            epsilon_dd_d[0] = 0.0;
            if(scattering_length != 0)
                epsilon_dd_d[0] = dipolar_length/scattering_length;
            Vtilde.reinit(nx,ny,nz);
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j)
                    for (int k = 0; k < nz; ++k)
                    {
                        double aux = TWOPI * (
                                kx[i]*sin(theta_mu)*cos(phi_mu) +
                                ky[j]*sin(theta_mu)*sin(phi_mu)+
                                kz[k]*cos(theta_mu));
                        double aux1 = TWOPI * sqrt(pow(kx[i], 2) + pow(ky[j], 2) + pow(kz[k], 2));
                        if (aux1 <= 1.E-6)
                            Vtilde(i,j,k) = -4*PI*scattering_length*epsilon_dd_d[0];
                        else
                            Vtilde(i,j,k) =
                                    12.0 * PI * scattering_length * epsilon_dd_d[0] * (pow(aux/aux1,2)-1.0/3.0);
                    }
            cudaMalloc(&Vtilde_d,npoints*sizeof(cuDoubleComplex));
            cudaMemcpy(Vtilde_d,Vtilde.data(),npoints*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
            cudaMalloc(&Phi_dd_d,npoints*sizeof(cuDoubleComplex));

            // Initialize gamma(\epsilon_dd) for the LHY correction
            cudaMallocManaged(&gamma_epsilondd_d,sizeof(double));
            gamma_epsilondd_d[0] = 0.0;
            if (epsilon_dd_d[0] != 0)
            {
                gamma_epsilondd_d[0] = 64.0*sqrt(PI)/3*sqrt(pow(scattering_length,5));
                double F_epsilon_dd=0.0;
                int n_theta=1000;
                double d_theta=PI/(n_theta-1);

                std::complex<double> csum;
                std::complex<double> caux;
                csum={0.0,0.0};
                for (int i = 0; i < n_theta; ++i)
                {
                    double theta=i*d_theta;
                    caux = pow(1.0+epsilon_dd_d[0]*(3.0*pow(cos(theta),2)-1.0),5);
                    caux = sqrt(caux);
                    csum += sin(theta)*caux;
                }
                csum *= d_theta;
                F_epsilon_dd = csum.real();
                gamma_epsilondd_d[0] *= F_epsilon_dd;
            }

            // Initialize the wave function for the output
            wave_function_output.reinit(nx,ny,nz);

        }

        /////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief Destructor frees device memory
         *
         * */
        /////////////////////////////////////////////////////////////////////////////////////

        DipolarGPSolver::~DipolarGPSolver()
        {
            cudaFree(external_potential_d);
            cudaFree(density_d);
            cudaFree(norm_d);
            cudaFree(initial_norm_d);
            cudaFree(wave_function_d);
            cudaFree(ft_wave_function_d);
            cudaFree(hpsi_d);
            cudaFree(x_axis_d);
            cudaFree(y_axis_d);
            cudaFree(z_axis_d);
            cudaFree(kx_axis_d);
            cudaFree(ky_axis_d);
            cudaFree(kz_axis_d);
            cudaFree(kmod2_d);
            cudaFree(r2mod_d);
            cudaFree(chemical_potential_d);
            cudaFree(scattering_length_d);
            cudaFree(alpha_d);
            cudaFree(beta_d);
            cudaFree(time_step_d);
            cudaFree(temporary_storage_d);
            cudaFree(Vtilde_d);
            cudaFree(Phi_dd_d);
            cudaFree(epsilon_dd_d);
            cudaFree(gamma_epsilondd_d);
        }


        /**
         * @brief Calculate the density profile
         *
         * */

        void DipolarGPSolver::calculate_density(double *density, cuDoubleComplex *wave_function,int size)
        {
            KernelUtilities::square_vector<<<gridSize,blockSize>>>(density,wave_function,size);
        }

        /////////////////////////////////////////////////////////////////////////////////////
        /**
         *
         * @brief Run the gradient descent
         *
         * No check of the residual! This is more barbaric!
         *
         *
         * */
        /////////////////////////////////////////////////////////////////////////////////////

        std::tuple<Vector<std::complex<double>>, double>
        DipolarGPSolver::run_gradient_descent(int max_num_iter,
                                              double alpha,
                                              double beta,
                                              std::ostream &output_stream,
                                              int write_output_every)

        {
            // Initialize the fft plan required for the calculation of the laplacian
            cufftHandle ft_plan;
            if(problem_is_2d)
                cufftPlan2d(&ft_plan,nx,ny,CUFFT_Z2Z);
            else if(problem_is_3d)
                cufftPlan3d(&ft_plan,nx,ny,nz,CUFFT_Z2Z);

            //--------------------------------------------------//
            //    Here the gradient-descent iterations start    //
            //--------------------------------------------------//

            // Allocate space for some new data on the device
            cudaMalloc(&alpha_d,sizeof(double));
            cudaMemcpy(alpha_d,&alpha,sizeof(double),cudaMemcpyHostToDevice);
            cudaMalloc(&beta_d,sizeof(double));
            cudaMemcpy(beta_d,&beta,sizeof(double),cudaMemcpyHostToDevice);
            cuDoubleComplex* psi_new;
            cuDoubleComplex* psi_old;
            cudaMalloc(&psi_new,npoints*sizeof(cuDoubleComplex));
            cudaMalloc(&psi_old,npoints*sizeof(cuDoubleComplex));
            cuDoubleComplex* c_density_d;
            cudaMalloc(&c_density_d,npoints*sizeof(cuDoubleComplex));

            // Loop starts here
            for (int it = 0; it < max_num_iter; ++it)
            {

                // Calculate the action of the laplacian
                cufftExecZ2Z(ft_plan, wave_function_d, ft_wave_function_d, CUFFT_FORWARD);
                cudaDeviceSynchronize();
                KernelUtilities::vector_multiplication<<<gridSize,blockSize>>>(ft_wave_function_d,kmod2_d,npoints);
                cudaDeviceSynchronize();
                cufftExecZ2Z(ft_plan, ft_wave_function_d, hpsi_d, CUFFT_INVERSE);
                cudaDeviceSynchronize();
                KernelUtilities::rescale<<<gridSize,blockSize>>>(hpsi_d,0.5*pow(TWOPI,2)/npoints,npoints);
                cudaDeviceSynchronize();

                // Calculate the dipolar potential
                KernelUtilities::square_vector<<<gridSize,blockSize>>>(c_density_d,wave_function_d,npoints);
                cudaDeviceSynchronize();
                cufftExecZ2Z(ft_plan,c_density_d,ft_wave_function_d,CUFFT_FORWARD);
                cudaDeviceSynchronize();
                KernelUtilities::vector_multiplication<<<gridSize,blockSize>>>(ft_wave_function_d,
                                                                               Vtilde_d,
                                                                               npoints);
                cudaDeviceSynchronize();
                cufftExecZ2Z(ft_plan,ft_wave_function_d,Phi_dd_d,CUFFT_INVERSE);
                cudaDeviceSynchronize();
                KernelUtilities::rescale<<<gridSize,blockSize>>>(Phi_dd_d,1./npoints,npoints);

                // Calculate the rest of H|psi>
                KernelUtilities::step_2_dipolar_hpsi<<<gridSize,blockSize>>>(hpsi_d,
                                                                             wave_function_d,
                                                                             external_potential_d,
                                                                             Phi_dd_d,
                                                                             scattering_length_d,
                                                                             gamma_epsilondd_d,
                                                                             npoints);
                cudaDeviceSynchronize();

                // Perform a gradient descent (plus heavy-ball) step
                KernelUtilities::gradient_descent_step<<<gridSize,blockSize>>>(wave_function_d,
                                                                               hpsi_d,
                                                                               psi_new,
                                                                               psi_old,
                                                                               alpha_d,
                                                                               beta_d,
                                                                               npoints);
                cudaDeviceSynchronize();

                // Normalize the wave function
                KernelUtilities::square_vector<<<gridSize,blockSize>>>(density_d,psi_new,npoints);
                cudaDeviceSynchronize();
                cub::DeviceReduce::Sum(temporary_storage_d,size_temporary_storage,density_d,norm_d,npoints);
                cudaDeviceSynchronize();
                norm_d[0] = norm_d[0]*dv;
                KernelUtilities::rescale<<<gridSize,blockSize>>>(wave_function_d,
                                                                 psi_new,
                                                                 sqrt(initial_norm_d[0]/norm_d[0]),
                                                                 npoints);
                cudaDeviceSynchronize();

                // Calculate the chemical potential
                KernelUtilities::vector_multiplication<<<gridSize,blockSize>>>(density_d,hpsi_d,wave_function_d,npoints);
                cudaDeviceSynchronize();
                cub::DeviceReduce::Sum(temporary_storage_d,size_temporary_storage,density_d,chemical_potential_d,npoints);
                cudaDeviceSynchronize();
                chemical_potential_d[0] = chemical_potential_d[0]*dv/norm_d[0];

                // Eventually write some output
                if(it % write_output_every == 0)
                    write_gradient_descent_output(it);

            }

            // Free the remaining arrays from the device
            cudaFree(psi_new);
            cudaFree(psi_old);
            cudaFree(c_density_d);

            // Copy out the results
            cudaMemcpy(result_wave_function.data(),
                       wave_function_d,
                       npoints*sizeof(cuDoubleComplex),
                       cudaMemcpyDeviceToHost);
            double result_chemical_potential = chemical_potential_d[0];

            // Return
            return std::make_pair(result_wave_function,result_chemical_potential);

        }

        /////////////////////////////////////////////////////////////////////////////////////
        /**
         *
         * @brief Write gradient descent output
         *
         * */
        /////////////////////////////////////////////////////////////////////////////////////

        void DipolarGPSolver::write_gradient_descent_output(int it)
        {
            std::cout << it << " " << chemical_potential_d[0] << std::endl;
        }

        /////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief Real-time operator splitting
         * */
        /////////////////////////////////////////////////////////////////////////////////////

        void DipolarGPSolver::run_operator_splitting(int number_of_time_steps,
                                                     double time_step,
                                                     std::ostream &output_stream,
                                                     int write_output_every)
        {
            // Copy input data into the device
            cudaMallocManaged(&time_step_d,sizeof(double));
            cudaMemcpy(time_step_d,&time_step,sizeof(double),cudaMemcpyHostToDevice);

            // Initialize the fft plan required for the calculation of the laplacian
            cufftHandle ft_plan;
            if(problem_is_2d)
                cufftPlan2d(&ft_plan,nx,ny,CUFFT_Z2Z);
            else if(problem_is_3d)
                cufftPlan3d(&ft_plan,nx,ny,nz,CUFFT_Z2Z);
            cuDoubleComplex* c_density_d;
            cudaMalloc(&c_density_d,npoints*sizeof(cuDoubleComplex));

            // Initialize other variables
            this->write_output_every=write_output_every;

            //----------------------------------------------------//
            //    Here the operator-splitting iterations start    //
            //----------------------------------------------------//
            for (size_t it = 0; it < number_of_time_steps; ++it)
            {

                // Write output starting from the very first iteration
                if(it % write_output_every == 0)
                    write_operator_splitting_output(it);

                // Calculate the current value of dipolar potential
                KernelUtilities::square_vector<<<gridSize,blockSize>>>(c_density_d,wave_function_d,npoints);
                cudaDeviceSynchronize();
                cufftExecZ2Z(ft_plan,c_density_d,ft_wave_function_d,CUFFT_FORWARD);
                cudaDeviceSynchronize();
                KernelUtilities::vector_multiplication<<<gridSize,blockSize>>>(ft_wave_function_d,
                                                                               Vtilde_d,
                                                                               npoints);
                cudaDeviceSynchronize();
                cufftExecZ2Z(ft_plan,ft_wave_function_d,Phi_dd_d,CUFFT_INVERSE);
                cudaDeviceSynchronize();
                KernelUtilities::rescale<<<gridSize,blockSize>>>(Phi_dd_d,1./npoints,npoints);
                cudaDeviceSynchronize();

                // Solve step-1 of operator splitting, i.e. the one NOT involving Fourier transforms
                KernelUtilities::step_1_operator_splitting_dipolars<<<gridSize,blockSize>>>(wave_function_d,
                                                                                            external_potential_d,
                                                                                            Phi_dd_d,
                                                                                            time_step_d,
                                                                                            scattering_length_d,
                                                                                            gamma_epsilondd_d,
                                                                                            npoints);
                cudaDeviceSynchronize();

                // Solve step-2 of operator splitting, i.e. the one actually involving Fourier transforms
                cufftExecZ2Z(ft_plan,wave_function_d,ft_wave_function_d,CUFFT_FORWARD);
                cudaDeviceSynchronize();
                KernelUtilities::aux_step_2_operator_splitting<<<gridSize,blockSize>>>(ft_wave_function_d,
                                                                                       kmod2_d,
                                                                                       time_step_d,
                                                                                       npoints);
                cudaDeviceSynchronize();
                cufftExecZ2Z(ft_plan,ft_wave_function_d,wave_function_d,CUFFT_INVERSE);
                cudaDeviceSynchronize();
                KernelUtilities::rescale<<<gridSize,blockSize>>>(wave_function_d,1./npoints,npoints);
                cudaDeviceSynchronize();

            }
            cudaFree(c_density_d);
        }

        /**
         *
         * @brief Operator splitting output
         *
         * */

        void DipolarGPSolver::write_operator_splitting_output(int it)
        {
            KernelUtilities::vector_average<<<gridSize,blockSize>>>(density_d,r2mod_d,wave_function_d,npoints);
            cudaDeviceSynchronize();
            double* x2m_d;
            cudaMallocManaged(&x2m_d,sizeof(double));
            cub::DeviceReduce::Sum(temporary_storage_d,size_temporary_storage,density_d,x2m_d,npoints);
            cudaDeviceSynchronize();
            x2m_d[0] = x2m_d[0]/norm_d[0];
            std::cout << it << " " << it*time_step_d[0] << " " << x2m_d[0] << std::endl;
            cudaFree(x2m_d);
        }

        /**
         *
         * @brief Reinit methods
         *
         * */

        void DipolarGPSolver::reinit(Vector<std::complex<double>> &psi, Vector<double> &Vext)
        {
            cudaMemcpy(wave_function_d,psi.data(),npoints*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
            cudaMemcpy(external_potential_d,Vext.data(),npoints*sizeof(double),cudaMemcpyHostToDevice);

        }

        void DipolarGPSolver::reinit(Vector<std::complex<double>> &psi, Vector<double> &Vext,double scattering_length)
        {
            cudaMemcpy(wave_function_d,psi.data(),npoints*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
            cudaMemcpy(external_potential_d,Vext.data(),npoints*sizeof(double),cudaMemcpyHostToDevice);
            cudaMemcpy(scattering_length_d,&scattering_length,sizeof(double),cudaMemcpyHostToDevice);
        }
    }
}
