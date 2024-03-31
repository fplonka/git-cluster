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
#include <filesystem>

#include "MetalSPE.hpp"

void testSPE()
{
    NS::AutoreleasePool *p_pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    // TODO: JFC
    MetalSPE *spe =
        new MetalSPE(std::filesystem::current_path().string() + "/dist_matrix_data");

    std::filesystem::path cwd = std::filesystem::current_path();

    spe->init_with_device(device);
    spe->prepare_data();
    spe->do_spe_loop();

    spe->write_results();

    p_pool->release();
}

int main(int argc, const char *argv[])
{
    // testAdder();
    testSPE();

    return 0;
}
