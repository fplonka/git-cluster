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

constant float epsilon = 0.0000001f;

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

bool is_cursed(float f) {
    return (fabs(f) < epsilon || fabs(f)  > 1.f / epsilon || isnan(f) || isinf(f));
}

kernel void update_coords_bfloat(
    device float2* coords,
    constant half* dist_matrix,
    constant uint& i, // pivot coord index
    constant float& learning_rate,
    constant uint& N,
    uint j [[thread_position_in_grid]])
{
    if (i == j)
        return;

    float2 xi = coords[i];
    float2 xj = coords[j];
    
    float rij = dist_matrix[i * N + j];
    if (isnan(rij))
        return;

    float dij = length(xi - xj);
    
    float2 update_dir = normalize(xi - xj);

    if (is_cursed(update_dir[0]) || is_cursed(update_dir[1]))
        return;
    
    if (is_cursed(learning_rate * (rij - dij)))
        return;

    coords[j] -= learning_rate * (rij - dij) * update_dir;
}

// kernel void update_coords_bfloat(
//     device float2* coords,
//     constant half* dist_matrix,
//     constant uint& i, // pivot coord index
//     constant float& learning_rate,
//     constant uint& N,
//     uint j [[thread_position_in_grid]])
// {
//     float2 xi = coords[i];
//     float2 xj = coords[j];
//     float rij = dist_matrix[i * N + j];
//     float dij = length(xi - xj);
//     // float dij = float(sqrt((xi[0] - xj[0])*(xi[0] - xj[0]) + (xi[1] - xj[1])*(xi[1] - xj[1])));
    
//     coords[j] -= (xi - xj) * learning_rate * (rij - dij) / (dij + epsilon);
// }


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