//
//  MetalAdder.cpp
//
//  Created by Srimukh Sripada on 04.12.21.
//

#include "MetalSPE.hpp"

#include <fstream>
#include <iostream>
#include <random>

void MetalSPE::init_params_from_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open dist matrix data file");
    }

    RunParams params{};

    file.read(reinterpret_cast<char*>(&params.num_iterations),
              sizeof(params.num_iterations));
    file.read(reinterpret_cast<char*>(&params.initial_lr),
              sizeof(params.initial_lr));
    file.read(reinterpret_cast<char*>(&params.final_lr),
              sizeof(params.final_lr));

    m_params = params;

    int N;
    file.read(reinterpret_cast<char*>(&N), sizeof(N));
    m_N = N;

    // Resize the matrix to N x N
    std::vector<std::vector<float>> matrix{};
    matrix.resize(N, std::vector<float>(N));

    // Read the array data
    for (auto& row : matrix) {
        file.read(reinterpret_cast<char*>(row.data()),
                  row.size() * sizeof(float));
    }
    m_dist_matrix_vec = matrix;
}

MetalSPE::MetalSPE(std::string distance_matrix_filename) {
    m_dist_matrix_vec = std::vector<std::vector<float>>();
    init_params_from_file(distance_matrix_filename);
    std::cout << "N is " << m_N << "\n";
    std::cout << "params are: " << m_params.num_iterations << " "
              << m_params.initial_lr << " " << m_params.final_lr << "\n";
}

void MetalSPE::init_with_device(MTL::Device* device) {
    m_device = device;
    NS::Error* error;
    auto default_library = m_device->newDefaultLibrary();

    if (!default_library) {
        std::cerr << "Failed to load default library.";
        std::exit(-1);
    }

    auto function_name =
        NS::String::string("update_coords", NS::ASCIIStringEncoding);
    auto spe_function = default_library->newFunction(function_name);

    if (!spe_function) {
        std::cerr << "Failed to find the update coords function.";
    }

    m_spe_function_pso =
        m_device->newComputePipelineState(spe_function, &error);
    m_command_queue = m_device->newCommandQueue();
};

void MetalSPE::prepare_data() {
    // Init with random coords
    m_coords_buffer = m_device->newBuffer(m_N * sizeof(float2),
                                          MTL::ResourceStorageModeShared);
    generate_random_float2_data(m_coords_buffer, m_N);

    // Place distance matrix in buffer
    size_t dist_matrix_size = m_N * m_N * sizeof(float);
    m_dist_matrix_buffer =
        m_device->newBuffer(dist_matrix_size, MTL::ResourceStorageModeShared);
    // std::memcpy(m_dist_matrix_buffer->contents(),
    // m_dist_matrix_nparr.data<float>(), dist_matrix_size);

    std::cout << "data prepared1\n";
    float* bufferPointer =
        static_cast<float*>(m_dist_matrix_buffer->contents());
    for (size_t i = 0; i < m_N; ++i) {
        std::memcpy(bufferPointer + i * m_N, m_dist_matrix_vec[i].data(),
                    m_N * sizeof(float));
    }

    std::cout << "data prepared\n";
}

// void MetalSPE::generate_random_float_data(MTL::Buffer* buffer) {
//     float* data_ptr = (float*)buffer->contents();
//     for (unsigned long index = 0; index < array_length; index++) {
//         data_ptr[index] = (float)rand() / (float)(RAND_MAX);
//     }
// }

void MetalSPE::generate_random_float2_data(MTL::Buffer* buffer,
                                           unsigned long num_elements) {
    float2* data_ptr = (float2*)buffer->contents();
    for (unsigned long index = 0; index < num_elements; index++) {
        data_ptr[index] = float2((float)rand() / (float)(RAND_MAX),
                                 (float)rand() / (float)(RAND_MAX));
    }
}

void MetalSPE::do_spe_loop() {
    std::cout << "starting\n";
    auto start = std::chrono::high_resolution_clock::now();
    float initial_lr = m_params.initial_lr;
    float final_lr = m_params.final_lr;
    uint num_iters = m_params.num_iterations;

    std::random_device rd;   // obtain a random number from hardware
    std::mt19937 gen(rd());  // seed the generator
    std::uniform_int_distribution<> distr(0, m_N);  // define the range

    const uint batch_size = 5000;
    uint batches_committed = 0;

    MTL::CommandBuffer* command_buffer = nullptr;
    while (batches_committed < num_iters) {
        command_buffer = m_command_queue->commandBuffer();
        for (int itr_idx = 0;
             itr_idx < batch_size && batches_committed < num_iters; ++itr_idx) {
            float lr = initial_lr + (final_lr - initial_lr) / float(num_iters) *
                                        float(batches_committed);
            uint pivot_idx = distr(gen);

            MTL::ComputeCommandEncoder* compute_encoder =
                command_buffer->computeCommandEncoder();
            encode_spe_command(compute_encoder, pivot_idx, lr);

            compute_encoder->endEncoding();
            batches_committed++;
        }
        command_buffer->commit();
        std::cout << "at " << batches_committed << "\n";
    }

    std::cout << "waiting...\n";
    command_buffer->waitUntilCompleted();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "SPE loop completed in " << elapsed.count() << " seconds\n";
}

void MetalSPE::encode_spe_command(MTL::ComputeCommandEncoder* compute_encoder,
                                  uint pivot_idx, float learning_rate) {
    compute_encoder->setComputePipelineState(m_spe_function_pso);

    // compute_encoder->setBuffer(m_coords_buffer, 0, 0);

    // compute_encoder->setBytes(&pivot_idx, sizeof(pivot_idx), 1);
    // compute_encoder->setBytes(&learning_rate, sizeof(learning_rate), 2);

    compute_encoder->setBuffer(m_coords_buffer, 0, 0);
    compute_encoder->setBuffer(m_dist_matrix_buffer, 0, 1);

    compute_encoder->setBytes(&pivot_idx, sizeof(pivot_idx), 2);
    compute_encoder->setBytes(&learning_rate, sizeof(learning_rate), 3);
    compute_encoder->setBytes(&m_N, sizeof(m_N), 4);

    MTL::Size grid_size = MTL::Size(m_N, 1, 1);

    NS::UInteger thread_group_size_ =
        m_spe_function_pso->maxTotalThreadsPerThreadgroup();

    if (thread_group_size_ > m_N) {
        thread_group_size_ = m_N;
    }

    MTL::Size thread_group_size = MTL::Size(thread_group_size_, 1, 1);

    compute_encoder->dispatchThreads(grid_size, thread_group_size);
}

void MetalSPE::write_results() {
    float2* bufferPointer = static_cast<float2*>(m_coords_buffer->contents());

    std::ofstream outputFile("embeddings.txt");
    for (int i = 0; i < m_N; ++i) {
        outputFile << bufferPointer[i].x << ", " << bufferPointer[i].y
                   << std::endl;
    }
    outputFile.close();
}
