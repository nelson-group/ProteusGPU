#ifndef VORONOI_H
#define VORONOI_H

#include <string>
#include "global/allvars.h"
#include "../knn/knn.h"
#include "../io/input.h"
#include "../io/output.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>

/* 
 * This part of the code is heavily inspired by the work of: Nicolas Ray, Dmitry Sokolov, 
 * Sylvain Lefebvre, Bruno L'evy, "Meshless Voronoi on the GPU", ACM Trans. Graph., 
 * vol. 37, no. 6, Dec. 2018. If you build upon this code, we recommend  
 * reading and citing their paper: https://doi.org/10.1145/3272127.3275092
 */ 

namespace voronoi {

    void compute_mesh(POINT_TYPE* pts_data, ICData& icData, InputHandler& input, OutputHandler& output);

    void compute_cells(int N_seedpts, knn_problem* knn);
    
    void cpu_compute_cell(int blocksPerGrid, int threadsPerBlock, int N_seedpts, POINT_TYPE* d_stored_points, unsigned int* d_knearests);

} // namespace voronoi

#endif // VORONOI_H
