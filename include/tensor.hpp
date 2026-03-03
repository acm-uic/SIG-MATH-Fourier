#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <cstdint>

template <typename T>
class Tensor 
{
    public:

        /* Constructors */
        Tensor() {}
        Tensor(std::vector<T> data_values, std::vector<uint32_t> shape_dimensions);
        Tensor(std::vector<uint32_t> shape);
        
        /* Metadata */
        size_t order() { return _shape.size();}
        size_t size() { return _data.size();}
        std::vector<uint32_t> shape() { return _shape;}

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
        void compute_strides();
        size_t size_from_shape(std::vector<uint32_t>& shape);
        size_t process_indices(std::vector<uint32_t>& indices);
};


#endif