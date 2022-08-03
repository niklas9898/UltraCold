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

#include "cudaDeviceVector.cuh"

namespace UltraCold{

    /**
     * @brief Implementation of constructor for a one-dimensional cudaVector
     *
     * */

    template<typename T>
    cudaDeviceVector<T>::cudaDeviceVector(int nx)
    {
        number_of_dimensions = 1;
        number_of_elements   = nx;
        extent_0 = nx;
        extent_1 = 0;
        extent_2 = 0;

        cudaMalloc(&elements,nx);
    }

    /**
     * @brief Implementation of constructor for a two-dimensional cudaVector
     *
     * */

    template<typename T>
    cudaDeviceVector<T>::cudaDeviceVector(int nx, int ny)
    {
        number_of_dimensions = 2;
        number_of_elements   = nx*ny;
        extent_0 = nx;
        extent_1 = ny;
        extent_2 = 0;

        cudaMalloc(&elements,nx*ny);

    }

    /**
     * @brief Implementation of constructor for a three-dimensional cudaVector
     *
     * */

    template<typename T>
    cudaDeviceVector<T>::cudaDeviceVector(int nx, int ny, int nz)
    {

        number_of_dimensions = 3;
        number_of_elements   = nx*ny*nz;
        extent_0 = nx;
        extent_1 = ny;
        extent_2 = nz;

        cudaMalloc(&elements,nx*ny*nz);
    }

    /**
     *
     * @brief Copy a given Vector
     *
     * */

    template<typename T>
    cudaDeviceVector<T>::cudaDeviceVector(Vector<T> host_vector)
    {
        number_of_dimensions = host_vector.order();
        number_of_elements   = host_vector.size();
        extent_0 = host_vector.extent(0);
        extent_1 = host_vector.extent(1);
        extent_2 = host_vector.extent(2);

        cudaMalloc(&elements,number_of_elements);

        cudaMemcpy(host_vector.data(),elements,number_of_elements,cudaMemcpyHostToDevice);

    }

    /**
     * @brief Implementation of destructor
     *
     * */

    template<typename T>
    cudaDeviceVector<T>::~cudaDeviceVector()
    {
        cudaFree(elements);
    }

    /**
     *
     * @brief Copy the content of the current vector out to a host vector
     *
     * */


    template<typename T>
    void cudaDeviceVector<T>::copy_out(Vector<T> host_vector)
    {

        cudaMemcpy(elements,host_vector.data(),number_of_elements,cudaMemcpyDeviceToHost);

    }

    /**
     * @brief Reinit method from a host Vector
     *
     * */

    template<typename T>
    void cudaDeviceVector<T>::reinit(Vector<T> host_vector)
    {

        number_of_dimensions = host_vector.order();
        number_of_elements   = host_vector.size();
        extent_0 = host_vector.extent(0);
        extent_1 = host_vector.extent(1);
        extent_2 = host_vector.extent(2);

        cudaMalloc(&elements,number_of_elements);

        cudaMemcpy(host_vector.data(),elements,number_of_elements,cudaMemcpyHostToDevice);

    }

    template<typename T>
    void cudaDeviceVector<T>::reinit(int nx)
    {
        number_of_dimensions = 1;
        number_of_elements   = nx;
        extent_0 = nx;
        extent_1 = 0;
        extent_2 = 0;

        cudaMalloc(&elements,nx);
    }

    template<typename T>
    void cudaDeviceVector<T>::reinit(int nx,int ny)
    {
        number_of_dimensions = 2;
        number_of_elements   = nx*ny;
        extent_0 = nx;
        extent_1 = ny;
        extent_2 = 0;

        cudaMalloc(&elements,number_of_elements);

    }

    template<typename T>
    void cudaDeviceVector<T>::reinit(int nx,int ny, int nz)
    {
        number_of_dimensions = 2;
        number_of_elements   = nx*ny*nz;
        extent_0 = nx;
        extent_1 = ny;
        extent_2 = nz;

        cudaMalloc(&elements,number_of_elements);

    }

    // Explicit template instantiations

    template class cudaDeviceVector<double>;
    template class cudaDeviceVector<std::complex<double>>;

}