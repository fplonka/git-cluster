//
//  add.metal
//
//  Created by Srimukh Sripada on 03.12.21.
//

#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] + inB[index];
}

constant float epsilon = 0.00000001f;

kernel void update_coords(
    device float2* coords,
    constant float* dist_matrix,
    constant uint& i, // pivot coord index
    constant float& learning_rate,
    constant uint& N,
    uint j [[thread_position_in_grid]])
{
    float2 xi = coords[i];
    float2 xj = coords[j];
    float rij = dist_matrix[i * N + j];

    float dij = sqrt((xi[0] - xj[0])*(xi[0] - xj[0]) + (xi[1] - xj[1])*(xi[1] - xj[1]));
    
    coords[j] -= (xi - xj) * learning_rate * (rij - dij) / (dij + epsilon);
}

kernel void update_coords_bfloat(
    device float2* coords,
    constant bfloat* dist_matrix,
    constant uint& i, // pivot coord index
    constant float& learning_rate,
    constant uint& N,
    uint j [[thread_position_in_grid]])
{
    float2 xi = coords[i];
    float2 xj = coords[j];
    float rij = dist_matrix[i * N + j];
    float dij = float(sqrt((xi[0] - xj[0])*(xi[0] - xj[0]) + (xi[1] - xj[1])*(xi[1] - xj[1])));
    
    coords[j] -= (xi - xj) * learning_rate * (rij - dij) / (dij + epsilon);
}

// kernel void update_coords(
//     device float2* coords,
//     constant uint& i, // pivot coord index
//     constant float& learning_rate,
//     uint j [[thread_position_in_grid]])
// {
//     float2 xi = coords[i];
//     float2 xj = coords[j];
//     float rij = 1.f;

//     float dij = sqrt((xi[0] - xj[0])*(xi[0] - xj[0]) + (xi[1] - xj[1])*(xi[1] - xj[1]));
    
//     coords[j] -= (xi - xj) * learning_rate * (rij - dij) / (dij + epsilon);
// }