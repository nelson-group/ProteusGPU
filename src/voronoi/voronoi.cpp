#include "voronoi.h"
#include "../global/allvars.h"
#include "../knn/knn.h"
#include "../io/input.h"
#include "../io/output.h"
#include <iostream>

namespace voronoi {

    void compute_mesh(POINT_TYPE* pts_data, ICData& icData, InputHandler& input, OutputHandler& output) {
        std::cout << "Computing Voronoi mesh..." << std::endl;

        // define knn problem
        knn_problem *knn = NULL;

        // prepare knn problem
        knn = knn::init((POINT_TYPE*) pts_data, icData.seedpos_dims[0]);

        // solve knn problem
        knn::solve(knn);

        #ifdef VERIFY
        if (!knn::verify(knn)) {exit(EXIT_FAILURE);}
        #endif

        // write KNN data to HDF5 file
        #if defined(USE_HDF5) && defined(WRITE_KNN_OUTPUT)
        POINT_TYPE* knn_pts = knn::get_points(knn);
        unsigned int* knn_nearest = knn::get_knearest(knn);
        unsigned int* knn_permutation = knn::get_permutation(knn);

        std::string knn_output_file = input.getParameter("output_knn_file");
        if (!output.writeKNNFile(knn_output_file, knn_pts, knn_nearest, knn_permutation, icData.seedpos_dims[0], _K_)) {exit(EXIT_FAILURE);}

        free(knn_pts);
        free(knn_nearest);
        free(knn_permutation);
        #endif

        // compute voronoi cells from knn results
        compute_cells(icData.seedpos_dims[0], knn);


        // free KNN resources
        knn::knn_free(&knn);
    }

    void compute_cells(int N_seedpts, knn_problem* knn) {

        int threadsPerBlock = _VORO_BLOCK_SIZE_;
        int blocksPerGrid = N_seedpts/threadsPerBlock + 1;

        cpu_compute_cell(blocksPerGrid, threadsPerBlock, N_seedpts, knn->d_stored_points, knn->d_knearests); // add stats and output later ... :D
    }

#ifdef CPU_DEBUG
    void cpu_compute_cell(int blocksPerGrid, int threadsPerBlock, int N_seedpts, POINT_TYPE* d_stored_points, unsigned int* d_knearests) {

        for (int blockId = 0; blockId < blocksPerGrid; blockId++) {
            for (int threadId = 0; threadId < threadsPerBlock; threadId++) {
                int seed_id = threadsPerBlock * blockId + threadId;
                if (seed_id >= N_seedpts) {break;}
                
                // compute a single voronoi cell in here...

                //create BBox
                //ConvexCell cc(seed, pts, &(gpu_stat[seed]));

                //FOR(v, _K_) {
	            //    unsigned int z = neigs[_K_ * seed + v];
                //    cc.clip_by_plane(z);
                
                //    if (cc.is_security_radius_reached(point_from_ptr3(pts + 3*z))) {
                //        break;
                //    }
                
                //    if (gpu_stat[seed] != success) {
                //        return;
                //    }
                //}
                // check security radius
                //if (!cc.is_security_radius_reached(point_from_ptr3(pts + 3 * neigs[_K_ * (seed+1) -1]))) {
                //    gpu_stat[seed] = security_radius_not_reached;
                //}

            }
        }
    }
#endif

} // namespace voronoi