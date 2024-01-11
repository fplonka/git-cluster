//
//  main.cpp
//
//  Created by Srimukh Sripada on 03.12.21.
//
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <iostream>

#include "MetalAdder.hpp"
#include "MetalSPE.hpp"

// void add_arrays(const float* inA, const float* inB, float* result, int
// length) {
//     for (int index = 0; index < length; index++) {
//         result[index] = inA[index] + inB[index];
//     }
// }

void testAdder() {
    NS::AutoreleasePool* p_pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MetalAdder* adder = new MetalAdder();
    adder->init_with_device(device);
    adder->prepare_data();
    adder->send_compute_command();

    std::cout << "Execution finished.";
    p_pool->release();
}

void testSPE() {
    NS::AutoreleasePool* p_pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MetalSPE* spe =
        new MetalSPE("/Users/filip/SQuaD-MDS/SQuaD MDS/dist_matrix_data");

    spe->init_with_device(device);
    spe->prepare_data();
    spe->do_spe_loop();

    std::cout << "Execution finished.";

    spe->write_results();

    p_pool->release();
}

int main(int argc, const char* argv[]) {
    // testAdder();
    testSPE();

    return 0;
}
