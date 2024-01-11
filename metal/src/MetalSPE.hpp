//
//  MetalAdder.hpp
//  metalnet
//
//  Created by Srimukh Sripada on 04.12.21.
//

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "cnpy.h"

struct float2 {
    float x, y;

    float2(float x, float y) : x(x), y(y) {}
};

struct RunParams {
    int num_iterations;
    float initial_lr;
    float final_lr;
    std::vector<std::vector<float>> matrix;  // Assuming the matrix is flattened
};

class MetalSPE {
   public:
    MetalSPE(std::string distancec_matrix_filename);

    MTL::Device *m_device;
    MTL::ComputePipelineState *m_spe_function_pso;
    MTL::CommandQueue *m_command_queue;

    MTL::Buffer *m_coords_buffer;
    MTL::Buffer *m_dist_matrix_buffer;

    void init_with_device(MTL::Device *);
    void prepare_data();
    void do_spe_loop();
    void write_results();

   private:
    void init_params_from_file(const std::string &filename);
    void generate_random_float_data(MTL::Buffer *buffer);
    void generate_random_float2_data(MTL::Buffer *buffer,
                                     unsigned long num_element);
    void encode_spe_command(MTL::ComputeCommandEncoder *compute_encoder,
                            uint pivot_idx, float learning_rate);

    RunParams m_params;
    unsigned long m_N;
    cnpy::NpyArray m_dist_matrix_nparr;
    std::vector<std::vector<float>> m_dist_matrix_vec;
};
