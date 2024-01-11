//
//  MetalAdder.cpp
//
//  Created by Srimukh Sripada on 04.12.21.
//

#include "MetalAdder.hpp"

#include <iostream>

const unsigned int array_length = 1 << 5;
const unsigned int buffer_size = array_length * sizeof(float);

void MetalAdder::init_with_device(MTL::Device* device) {
    m_device = device;
    NS::Error* error;
    auto default_library = m_device->newDefaultLibrary();

    if (!default_library) {
        std::cerr << "Failed to load default library.";
        std::exit(-1);
    }

    auto function_name =
        NS::String::string("add_arrays", NS::ASCIIStringEncoding);
    auto add_function = default_library->newFunction(function_name);

    if (!add_function) {
        std::cerr << "Failed to find the adder function.";
    }

    m_add_function_pso =
        m_device->newComputePipelineState(add_function, &error);
    m_command_queue = m_device->newCommandQueue();
};

void MetalAdder::prepare_data() {
    m_buffer_A =
        m_device->newBuffer(buffer_size, MTL::ResourceStorageModeShared);
    m_buffer_B =
        m_device->newBuffer(buffer_size, MTL::ResourceStorageModeShared);
    m_buffer_result =
        m_device->newBuffer(buffer_size, MTL::ResourceStorageModeShared);

    generate_random_float_data(m_buffer_A);
    generate_random_float_data(m_buffer_B);
}

void MetalAdder::generate_random_float_data(MTL::Buffer* buffer) {
    float* data_ptr = (float*)buffer->contents();
    for (unsigned long index = 0; index < array_length; index++) {
        data_ptr[index] = (float)rand() / (float)(RAND_MAX);
    }
}

// void MetalAdder::generate_random_float2_data(MTL::Buffer* buffer,
//                                              unsigned long num_elements) {
//     float2* data_ptr = (float2*)buffer->contents();
//     for (unsigned long index = 0; index < num_elements; index++) {
//         // Generate random coordinates between 0 and 1
//         data_ptr[index] = float2((float)rand() / (float)(RAND_MAX),
//                                  (float)rand() / (float)(RAND_MAX));
//     }
// }

void MetalAdder::send_compute_command() {
    MTL::CommandBuffer* command_buffer = m_command_queue->commandBuffer();
    //    assert(command_buffer != nullptr);
    MTL::ComputeCommandEncoder* compute_encoder =
        command_buffer->computeCommandEncoder();
    encode_add_command(compute_encoder);
    compute_encoder->endEncoding();
    //    MTL::CommandBufferStatus status = command_buffer->status();
    //    std::cout << status << std::endl;
    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    verify_results();
}

void MetalAdder::encode_add_command(
    MTL::ComputeCommandEncoder* compute_encoder) {
    compute_encoder->setComputePipelineState(m_add_function_pso);
    compute_encoder->setBuffer(m_buffer_A, 0, 0);
    compute_encoder->setBuffer(m_buffer_B, 0, 1);
    compute_encoder->setBuffer(m_buffer_result, 0, 2);

    MTL::Size grid_size = MTL::Size(array_length, 1, 1);

    NS::UInteger thread_group_size_ =
        m_add_function_pso->maxTotalThreadsPerThreadgroup();

    std::cout << "thread group size: " << thread_group_size_ << "\n";
    if (thread_group_size_ > array_length) {
        thread_group_size_ = array_length;
    }

    MTL::Size thread_group_size = MTL::Size(thread_group_size_, 1, 1);

    compute_encoder->dispatchThreads(grid_size, thread_group_size);
}

// void MetalAdder::encode_spe_command(MTL::ComputeCommandEncoder*
// compute_encoder,
//                                     float2 pivot, float learning_rate) {
//     compute_encoder->setComputePipelineState(m_add_function_pso);  // TODO:

//     compute_encoder->setBuffer(m_buffer_A, 0, 0);  // TODO:

//     compute_encoder->setBytes(&pivot, sizeof(pivot), 1);
//     compute_encoder->setBytes(&learning_rate, sizeof(learning_rate), 2);

//     MTL::Size grid_size = MTL::Size(array_length, 1, 1);

//     NS::UInteger thread_group_size_ =
//         m_add_function_pso->maxTotalThreadsPerThreadgroup();

//     std::cout << "thread group size: " << thread_group_size_ << "\n";
//     if (thread_group_size_ > array_length) {
//         thread_group_size_ = array_length;
//     }

//     MTL::Size thread_group_size = MTL::Size(thread_group_size_, 1, 1);

//     compute_encoder->dispatchThreads(grid_size, thread_group_size);
// }

void MetalAdder::verify_results() {
    auto a = (float*)m_buffer_A->contents();
    auto b = (float*)m_buffer_B->contents();
    auto result = (float*)m_buffer_result->contents();

    for (unsigned long index = 0; index < array_length; index++) {
        if (result[index] != (a[index] + b[index])) {
            std::cout << "Comput ERROR: index=" << index
                      << "result=" << result[index] << "vs "
                      << a[index] + b[index] << "=a+b\n";
            assert(result[index] == (a[index] + b[index]));
        }
    }
    std::cout << "Compute results as expected\n";
}
