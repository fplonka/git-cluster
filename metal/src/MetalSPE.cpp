//
//  MetalAdder.cpp
//
//  Created by Srimukh Sripada on 04.12.21.
//

#include "MetalSPE.hpp"

#include <math.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

MetalSPE::MetalSPE(std::string distance_matrix_filename)
    : m_params_filename(distance_matrix_filename) {}

void MetalSPE::init_with_device(MTL::Device* device) {
    m_device = device;
    NS::Error* error;
    auto default_library = m_device->newDefaultLibrary();

    if (!default_library) {
        std::cerr << "Failed to load default library.";
        std::exit(-1);
    }

    // auto function_name =
    //     NS::String::string("update_coords", NS::ASCIIStringEncoding);
    auto function_name =
        NS::String::string("update_coords_bfloat", NS::ASCIIStringEncoding);
    auto spe_function = default_library->newFunction(function_name);

    if (!spe_function) {
        std::cerr << "Failed to find the update coords function.";
    }

    m_spe_function_pso =
        m_device->newComputePipelineState(spe_function, &error);
    m_command_queue = m_device->newCommandQueue();
};

bool isHalfNaN(uint16_t value) {
    // Mask for the exponent (5 bits set to 1, shifted left 10 positions)
    const uint16_t exponentMask = 0x7C00;  // 0b0111110000000000

    // Mask for the fraction (10 bits set to 1)
    const uint16_t fractionMask = 0x03FF;  // 0b0000001111111111

    // Check for NaN: exponent is all 1s and fraction is non-zero
    return (value & exponentMask) == exponentMask &&
           (value & fractionMask) != 0;
}

void MetalSPE::prepare_data() {
    // Read run data and params from file
    std::cout << "Reading matrix data in C++ subprocess...\n";
    std::cout.flush();
    std::ifstream file(m_params_filename, std::ios::binary);

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

    // Init with random coords
    m_coords_buffer = m_device->newBuffer(m_N * sizeof(float2),
                                          MTL::ResourceStorageModeShared);
    generate_random_float2_data(m_coords_buffer, m_N);

    size_t dist_matrix_size = m_N * m_N * sizeof(uint16_t);
    m_dist_matrix_buffer =
        m_device->newBuffer(dist_matrix_size, MTL::ResourceStorageModeShared);
    if (!m_dist_matrix_buffer) {
        std::cout << "couldn't make buffer\n...";
    }
    file.read(reinterpret_cast<char*>(m_dist_matrix_buffer->contents()),
              dist_matrix_size);

    // TEST READ
    const uint16_t* halfBuffer =
        reinterpret_cast<const uint16_t*>(m_dist_matrix_buffer->contents());

    for (size_t i = 0; i < m_N * m_N; ++i) {
        if (isHalfNaN(halfBuffer[i])) {
            std::cout << "NaN found at index " << i << std::endl;
        }
    }

    // std::cout << "checking vals!\n";
    // uint16_t* bufferPointer =
    //     static_cast<uint16_t*>(m_dist_matrix_buffer->contents());
    // for (int i = 0; i < m_N; ++i) {
    //      val = bufferPointer[i];
    //     if (val < 0.f || val > 1.f) {
    //         std::cout << "WEIRD VAL AT " << i << ": " << val << "\n";
    //     }
    // }
}

void MetalSPE::generate_random_float2_data(MTL::Buffer* buffer,
                                           unsigned long num_elements) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float2* data_ptr = (float2*)buffer->contents();
    for (unsigned long index = 0; index < num_elements; index++) {
        data_ptr[index] = float2(dist(rng), dist(rng));
    }
}

void MetalSPE::do_spe_loop() {
    auto start = std::chrono::high_resolution_clock::now();
    float initial_lr = m_params.initial_lr;
    float final_lr = m_params.final_lr;
    uint num_iters = m_params.num_iterations;

    std::random_device rd;   // obtain a random number from hardware
    std::mt19937 gen(rd());  // seed the generator
    std::uniform_int_distribution<> distr(0, m_N);  // define the range

    const uint batch_size = 5000;
    uint batches_committed = 0;

    std::cout << std::fixed << std::setprecision(2);
    std::cerr << std::fixed << std::setprecision(2);

    std::cerr << "\n";
    MTL::CommandBuffer* command_buffer = nullptr;
    while (batches_committed < num_iters) {
        command_buffer = m_command_queue->commandBuffer();
        for (int itr_idx = 0;
             itr_idx < batch_size && batches_committed < num_iters; ++itr_idx) {
            float lr =
                final_lr + 0.5f * (initial_lr - final_lr) *
                               (1.f + cos(M_PI * ((float)batches_committed) /
                                          float(num_iters)));

            // float lr = initial_lr + (final_lr - initial_lr) /
            // float(num_iters) *
            //                             float(batches_committed);
            uint pivot_idx = distr(gen);

            MTL::ComputeCommandEncoder* compute_encoder =
                command_buffer->computeCommandEncoder();
            encode_spe_command(compute_encoder, pivot_idx, lr);

            compute_encoder->endEncoding();
            batches_committed++;
        }
        command_buffer->commit();
        std::cerr << "\rCommmmited "
                  << ((float)batches_committed / (float)num_iters * 100.f)
                  << "% of all GPU commands...";
        // std::cout.flush();
        // std::cout << "at " << batches_committed << "\n";
    }
    // std::cout << "\n";

    // std::cout << "waiting...\n";
    command_buffer->waitUntilCompleted();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // std::cout << "Computing embeddings took " << elapsed.count()
    //   << " seconds\n";
    // std::cout.flush();
}

void MetalSPE::encode_spe_command(MTL::ComputeCommandEncoder* compute_encoder,
                                  uint pivot_idx, float learning_rate) {
    compute_encoder->setComputePipelineState(m_spe_function_pso);

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

    std::ofstream outputFile("metal/embeddings.txt");
    for (int i = 0; i < m_N; ++i) {
        outputFile << bufferPointer[i].x << ", " << bufferPointer[i].y
                   << std::endl;
    }
    outputFile.close();
}
