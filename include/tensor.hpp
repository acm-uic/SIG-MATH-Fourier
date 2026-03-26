#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <cstdint>

// Memory access layout pattern for the Tensor
enum class Layout {LEFT, RIGHT};

// Tensor Class
template <typename T>
class Tensor 
{
    public:
        /* Constructors */
        Tensor() {}
        Tensor(std::vector<T> data_values, std::vector<uint32_t> shape);
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
        void compute_strides(Layout layout=Layout::RIGHT);
        uint64_t size_from_shape(std::vector<uint32_t>& shape);
        uint64_t process_indices(std::vector<uint32_t>& indices);
};


/*
*   Constructors
*/
template<typename T>
Tensor<T>::Tensor(std::vector<T> data_values, std::vector<uint32_t> shape)
    : _data(data_values)
    , _shape(shape)
{
    // Bounds checking
    if (size_from_shape(shape) != data_values.size()) {
        throw std::invalid_argument("Invalid shapes with data values size.");
    }
    // Setting strides
    compute_strides();
}

template<typename T>
Tensor<T>::Tensor(std::vector<uint32_t> shape)
    : _shape(shape)
{
    uint64_t total_size = size_from_shape(_shape);
     _data.resize(total_size, T() );
    compute_strides();
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
            throw std::out_of_range("Index " + std::to_string(k) + " out of bounds.");
        }

        // Update flattened index if everything is good
        flattened_index += indices[k] * _strides[k];
    }

    return flattened_index;
}

// Computing the data size from shape
template<typename T>
uint64_t Tensor<T>::size_from_shape(std::vector<uint32_t>& shape)
{
    // Empty shape means empty size
    uint32_t order = shape.size();
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
void Tensor<T>::compute_strides(Layout layout)
{
    // Re-configure strides based on shape
    uint32_t order = _shape.size();
    _strides.resize(order);

    switch (layout) 
    {
        case (Layout::RIGHT):
            // Right layout strides
            uint64_t stride_val = 1;
            for (int64_t k = order - 1; k >= 0; k--) {
                _strides[k] = stride_val;
                stride_val *= _shape[k];
            }

            break;
        
        case (Layout::LEFT):
            // Left layout strides
            uint64_t stride_val = 1;
            for (uint32_t k = 0; k < order; k++) {
                _strides[k] = stride_val;
                stride_val *= _shape[k];
            }
            break;

        default:
            throw std::invalid_argument("Unknown Layout Pattern!");
            break;
    }
}

/*
*   Exposed accessors methods
*/
template<typename T>
T Tensor<T>::value_at(std::vector<uint32_t>& indices)
{
    unsigned int flattened_index = process_indices(indices);
    return _data[flattened_index];
}

template<typename T>
std::vector<T> Tensor<T>::extract_1d_slice(uint32_t dim, std::vector<uint32_t>& indices)
{
    // Setting up the slice
    uint32_t slice_size = _shape[dim];
    std::vector<T> slice(slice_size);

    // Retrieve slice data
    for (unsigned int k = 0; k < slice_size; k++) {
        indices[dim] = k;
        slice[k] = value_at[indices];
    }

    return slice;
}


/*
*   Exposed setters
*/
template<typename T>
void Tensor<T>::set_value_at(T val, std::vector<uint32_t>& indices)
{
    unsigned int flattened_index = process_indices(indices);
    _data[flattened_index] = val;
}

template<typename T>
void Tensor<T>::set_1d_slice(uint32_t dim, std::vector<uint32_t> indices, std::vector<T>& new_values)
{
    uint32_t slice_size = _shape.at(dim);
    for (uint32_t k = 0; k < slice_size; k++) {
        indices[dim] = k;
        set_value_at(new_values[k], indices)
    }
}

#endif