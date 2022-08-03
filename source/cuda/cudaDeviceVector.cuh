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

#ifndef ULTRACOLD_CUDA_DEVICE_VECTOR
#define ULTRACOLD_CUDA_DEVICE_VECTOR

#include <complex>
#include "DataOut.hpp"
#include "Vector.hpp"

namespace UltraCold{

    /**
     * @brief Class to handle cuda vectors
     * @author Santo Maria Roccuzzo (santom.roccuzzo@gmail.com)
     *
     * This is a simple class that basically allows to hide allocation and release of device memory inside a class,
     * in order to avoid potential issues coming from naked pointers.
     *
     * */

    template <typename T>
    class cudaDeviceVector
    {

        public:

            // Constructors

            cudaDeviceVector() = default; // Default constructor
            cudaDeviceVector(int); // Constructor for a one-dimensional vector
            cudaDeviceVector(int, int); // Constructor for a two-dimensional vector
            cudaDeviceVector(int, int, int); // Constructor for a three-dimensional vector
            cudaDeviceVector(Vector<T>); // Constructor for copying in from a given Vector

            ~cudaDeviceVector(); // Release cuda-allocated memory

            void copy_out(Vector<T>); // Copy the content of this vector to a host Vector

            // Reinit methods

            void reinit(Vector<T>);
            void reinit(int);
            void reinit(int,int);
            void reinit(int,int,int);

        private:

            T* elements = nullptr;      // Pointer to the first element of the Vector
            int number_of_dimensions;   // Order of the Vector, i.e. the number of its dimensions
            int number_of_elements;     // Total number of elements
            int extent_0;               // Number of elements along direction 0
            int extent_1;               // Number of elements along direction 1
            int extent_2;               // Number of elements along direction 2

    };
}


#endif // ULTRACOLD_CUDA_DEVICE_VECTOR