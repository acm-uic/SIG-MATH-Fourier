#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <cstdint>

/*
*   Memory access layout pattern for the Tensor
*/
enum class Layout 
{
    LEFT,
    RIGHT
};

template <typename T>
class Tensor 
{
    public:

        /* Constructors */
        Tensor() {}
        Tensor(std::vector<T> data_values, std::vector<uint32_t> shape_dimensions);
        Tensor(std::vector<uint32_t> shape);
        
        /* Metadata */
        uint64_t order()  { return _shape.size();}
        uint64_t size()   { return _data.size();}
        std::vector<uint32_t> shape()   { return _shape;}

        /* Setters */
        void set_value_at(T val, std::vector<uint32_t>& indices);
        void set_1d_slice(uint32_t dim, std::vector<uint32_t> indices, std::vector<T>& new_values);
        
        /* Accessors */
        T value_at(std::vector<uint32_t>& indices);
        std::vector<T> extract_1d_slice(uint32_t dim, std::vector<uint32_t>& indices);

    private: 

        /* Data fields */
        std::vector<T> _data;
        std::vector<uint32_t> _strides;
        std::vector<uint32_t> _shape;

        /* Helper methods */
        void compute_strides(Layout layout);
        uint64_t size_from_shape(std::vector<uint32_t>& shape);
        uint64_t process_indices(std::vector<uint32_t>& indices);
};


/*
*   Constructors
*/
template<typename T>
Tensor<T>::Tensor(std::vector<T> data_values, std::vector<uint32_t> shape_dimensions)
{

}

/*
*   Private helper methods
*/

// Mapping the given accessing indices to the flattened data index
template<typename T>
uint64_t Tensor<T>::process_indices(std::vector<uint32_t>& indices)
{
    // Checking if the indexing orders matches up with shape orders
    uint32_t order = _shape.size();
    if (order ! = indices.size()) {
        throw std::invalid_argument("Invalid indices order!");
    }

    uint64_t flattened_index = 0;
    for (uint32_t k = 0; k < order; k++) {
        
        // Index bounds check
        if (indices[k] >= _shape[k]) {
            throw std::out_of_range("Index %d out of bounds", k);
        }

        // Update flattened index if everything is good
        flattened_index += indices[k] * _strides[k];
    }
}

// 
template<typename T>
uint64_t Tensor<T>::size_from_shape(std::vector<uint32_t>& shape)
{
    // Empty shape means empty size
    uint32_t order = 0;
    if (order == 0)
        return 0;

    // Update the total size based on shape
    uint64_t total_size = 1;
    for (uint32_t k = 0; k < order; k++) {
        total_size *= shape[k];
    }

    return total_size;
}

// Computing strides for the Tensor depend on the layout
// Default layout is RIGHT-most (Row-major C layout)
template<typename T>
void Tensor<T>::compute_strides(Layout layout=Layout::RIGHT)
{
    // Re-configure strides based on shape
    uint32_t order = _shape.size();
    _strides.reshape(order);

    switch layout 
    {
        case (Layout::RIGHT):

            // Right layout strides
            uint64_t stride_val = 1;
            for (uint32_t k = order - 1; k >= 0; k--) {
                _strides.at(k) = stride_val;
                stride_val *= _shape.at(k);
            }

            break;
        
        case (Layout::LEFT):
        
            // Left layout strides
            uint64_t stride_val = 1;
            for (uint32_t k = 0; k < order; k++) {
                _strides.at(k) = stride_val;
                stride_val *= _shape.at(k);
            }
            break;

        default:
            throw std::invalid_argument("Unknown Layout Pattern");
            break;
    }
}


#endif