// Wrapper header for sketch2model.h
// This ensures Eigen macros are defined before sketch2model.h includes Eigen

#ifndef TEDDY_SKETCH2MODEL_WRAPPER_H
#define TEDDY_SKETCH2MODEL_WRAPPER_H

// CRITICAL: Define Eigen macros BEFORE including any Eigen headers
#ifndef EIGEN_DONT_INCLUDE_ITERATIVELINEARSOLVERS
#define EIGEN_DONT_INCLUDE_ITERATIVELINEARSOLVERS
#endif

#ifndef EIGEN_SPARSESOLVERBASE_H
#define EIGEN_SPARSESOLVERBASE_H
#endif

// CRITICAL: Include order matters!
// 1. First include Eigen/Core (needed by SparseSolverBase-mod.h)
#include <Eigen/Core>

// 2. Then include SparseSolverBase-mod.h (needed before Eigen/Sparse)
#include "../eigen3-mod/Eigen/src/SparseCore/SparseSolverBase-mod.h"

// 3. Now include sketch2model.h (which includes Eigen/Sparse that needs SparseSolverBase)
#include "../src/sketch2model.h"

#endif // TEDDY_SKETCH2MODEL_WRAPPER_H

