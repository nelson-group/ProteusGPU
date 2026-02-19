#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>
#include "global/allvars.h"
#include "io/input.h"
#include "io/output.h"
#include "knn/knn.h"
#include "begrun/begrun.h"

int main(int argc, char* argv[]) {
    
    // welcome
    begrun::print_banner();

    // load param.txt
    InputHandler input = begrun::loadInputFiles(argc, argv);
    
    // initalize output handler
    std::string outputDir = input.getParameter("output_directory");
    OutputHandler output(outputDir);
    if (!output.initialize()) return EXIT_FAILURE;

    // read IC file
    ICData icData;
    if (!input.readICFile(input.getParameter("ic_file"), icData)) {return EXIT_FAILURE;}

    std::vector<double> pts = icData.seedpos;


    // -------- actual code starts here --------
    _K_ = input.getParameterInt("knn_k");
    _boxsize_ = input.getParameterDouble("box_size");
    _KNN_BLOCK_SIZE_ = input.getParameterInt("knn_block_size");

    // define knn problem
    knn_problem *knn = NULL;

    // prepare knn problem
    knn = knn::init((double3*) pts.data(), icData.seedpos_dims[0]);


    // solve knn problem
    knn::solve(knn);

    double3* knn_pts = knn::get_points(knn);
    unsigned int* knn_nearest = knn::get_knearest(knn);
    unsigned int* knn_permutation = knn::get_permutation(knn);

    // -------- testing area --------
    // write KNN data to HDF5 file
#if defined(USE_HDF5) && defined(WRITE_KNN_OUTPUT)
    std::string knn_output_file = input.getParameter("output_knn_file");
    if (!output.writeKNNFile(knn_output_file, knn_pts, knn_nearest, knn_permutation, icData.seedpos_dims[0], _K_)) {return EXIT_FAILURE;}
#endif

    // free KNN resources
    free(knn_pts);
    free(knn_nearest);
    free(knn_permutation);
    knn::knn_free(&knn);

    // write mesh file
    //MeshCellData meshData;
    // here meshData would have to be filled with actual data
    //if (!output.writeMeshFile(input.getParameter("output_mesh_file"), meshData)) {return EXIT_FAILURE;}

    knn::printInfo();
    return 0;
}
