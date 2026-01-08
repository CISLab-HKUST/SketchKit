// Copyright 2020-2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// CRITICAL: Define Eigen macros BEFORE including ANY Eigen headers
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

// 3. Now we can safely include sketch2model.h (which includes Eigen/Sparse)
#include "sketch2model.h"

// 4. Include libigl headers (which may also include Eigen headers)
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/massmatrix.h>
#include <igl/min_quad_with_fixed.h>
#include <image/imageUtils.h>
#include <ir3d-utils/MeshBuilder.h>
#include <ir3d-utils/regionToMesh.h>

// Eigen/Core and Eigen/Sparse are already included via sketch2model.h
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

#include "common/commonStructs.h"
#include "common/macros.h"
#include <miscutils/macros.h>

using namespace std;
using namespace Eigen;
using namespace igl;

#define ENABLE_MERGING_COMMON_BOUNDARY_IN_PREPROCESS

template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

bool sketch2model(
    const std::vector<Imguc> &outlineImgs,
    const std::vector<Imguc> &regionImgs,
    Eigen::MatrixXd &outputVertices,
    Eigen::MatrixXi &outputFaces,
    const std::string &triangleOpts,
    const int subsFactor,
    const int smoothFactor,
    const double defaultInflationAmount,
    const std::vector<double> &regionInflationAmounts) {
  
  // Check input
  if (outlineImgs.empty() || regionImgs.empty()) {
    std::cerr << "Error: Empty input images" << std::endl;
    return false;
  }
  
  if (outlineImgs.size() != regionImgs.size()) {
    std::cerr << "Error: Number of outline images != number of region images" << std::endl;
    return false;
  }

  const int N = regionImgs.size();
  const int numRegions = 2 * N;  // Front and back surfaces

  // ========== Step 1: Convert 2D regions to planar meshes (following original implementation) ==========
  vector<MatrixXd> VsFront, VsBack;
  vector<MatrixXi> FsFront, FsBack;
  vector<vector<vector<Vector2f>>> regionsBnds;
  const bool interiorMergingPtsOnly = true;
  std::set<int> mergeBothSides;  // Empty, meaning don't merge both sides

  // Front surface mesh
  vector<TriData> triData;
  outlineToMesh(outlineImgs, regionImgs, true, false, interiorMergingPtsOnly,
                triangleOpts, mergeBothSides, VsFront, FsFront, smoothFactor,
                regionsBnds, triData);
  
  // Back surface mesh
  outlineToMesh(outlineImgs, regionImgs, false, true, interiorMergingPtsOnly,
                triangleOpts, mergeBothSides, VsBack, FsBack, smoothFactor);

  // Create boundary point indices (from regionsBnds)
  vector<vector<vector<int>>> customBnds;
  forlist(k, regionsBnds) {
    vector<vector<int>> bnd1;
    int c = 0;
    forlist(l, regionsBnds[k]) {
      vector<int> bnd2;
      forlist(m, regionsBnds[k][l]) {
        bnd2.push_back(c);
        c++;
      }
      bnd1.push_back(bnd2);
    }
    customBnds.push_back(bnd1);
  }

  // ========== Step 2: Process all planar meshes and merge (following original implementation) ==========
  const int Z_COORD = 2;
  MeshBuilder mb;
  mb.graftingBndStitching = false;
  mb.graftingBndIneq = false;
  mb.flipReg.resize(numRegions, false);

  // Use processTwoSidedMesh to process front and back surfaces (following original implementation)
  forlist(i, regionImgs) {
#ifdef ENABLE_MERGING_COMMON_BOUNDARY_IN_PREPROCESS
    mb.processTwoSidedMesh(VsFront[i], FsFront[i], VsBack[i], FsBack[i], false,
                           customBnds[i], true);
#else
    // If preprocessing merge is not enabled, process front and back surfaces separately
    mb.processPlanarMesh(VsFront[i], FsFront[i], false, customBnds[i], false);
    mb.processPlanarMesh(VsBack[i], FsBack[i], false, customBnds[i], false);
#endif
  }
  
  // Set default inflation amounts (handle front and back surfaces separately, following original implementation)
  vector<float> inflationAmount(numRegions, defaultInflationAmount);
  forlist(i, regionImgs) {
    // Set back surface flip flag
    mb.flipReg[2 * i + 1] = true;
    
    // Use provided inflation amounts (if available)
    if (!regionInflationAmounts.empty() && i < regionInflationAmounts.size()) {
      inflationAmount[2 * i] = inflationAmount[2 * i + 1] = regionInflationAmounts[i];
    }
    // Back surface takes negative value (as in original implementation)
    inflationAmount[2 * i + 1] *= -1;
  }
  
  // Create complete mesh (must call)
  mb.createCompleteMesh();
  vector<vector<int>> bnds = mb.bnds;

  // Get complete mesh and boundary
  MatrixXd V;
  MatrixXi F;
  vector<int> bnd, neumannBnd;
  mb.getCompleteMesh(V, F);
  mb.getCompleteBoundary(bnd, neumannBnd);

  // ========== Step 3: 3D inflation (compute Z coordinates) following original implementation ==========
  
  // Compute mass matrix (using VORONOI type, consistent with original implementation)
  SparseMatrix<double> MNotMerged;
  massmatrix(V, F, MASSMATRIX_TYPE_VORONOI, MNotMerged);
  
  // Check validity of mass matrix (check from original implementation)
  if (MNotMerged.nonZeros() - MNotMerged.rows() != 0) {
    std::cerr << "Error: M.nonZeros()-M.rows()=" << (MNotMerged.nonZeros() - MNotMerged.rows())
              << " probably a problem in interconnection (non-manifold mesh?)" << std::endl;
    return false;
  }
  
  // Compute inverse of mass matrix
  SparseMatrix<double> MinvNotMerged;
  invert_diag(MNotMerged, MinvNotMerged);
  
  // Compute planar Laplacian matrix
  SparseMatrix<double> LFlatNotMerged;
  cotmatrix(V, F, LFlatNotMerged);

  // Set boundary conditions: boundary points have Z coordinate = 0 (planar)
  const int nBnd = bnd.size();
  const int nV = V.rows();
  VectorXi b(nBnd);
  VectorXd bc(nBnd);
  VectorXd inB(nV);
  VectorXd outZ;
  
  forlist(i, bnd) {
    b(i) = bnd[i];
  }
  bc.fill(0.0);  // Boundary stays on the plane

  // Set initial Z values for interior points (based on inflation amount) - original implementation uses -inflationAmount
  // Note: In original implementation, inflationAmount is vector<float>, containing inflation amounts for front and back surfaces
  for (int i = 0; i < mb.getMeshesCount(); i++) {
    if (i < inflationAmount.size()) {
      float inflAmt = inflationAmount[i];
      forlist(j, mb.parts1[i]) {
        int ind = mb.parts1[i][j];
        inB(ind) = -inflAmt;  // Original implementation uses negative value
      }
    }
  }

  // Solve optimization problem: minimize Laplacian energy with boundary constraints
  // Original implementation: Q = Minv * LFlat, directly uses inB as right-hand side
  SparseMatrix<double> Q = MinvNotMerged * LFlatNotMerged;
  min_quad_with_fixed(Q, inB, b, bc, SparseMatrix<double>(), VectorXd(), false,
                      outZ);

  // Apply Z coordinates: use sign and square root (maintaining original implementation method)
  forlist(i, mb.parts1) {
    forlist(j, mb.parts1[i]) {
      int ind = mb.parts1[i][j];
      V(ind, Z_COORD) = sgn(outZ(ind)) * sqrt(abs(outZ(ind)));
    }
  }

  MatrixXd VPreinf = V;

  // ========== Step 4: Post-processing: vertex merging and final mesh generation ==========
  
  // Compute Laplacian matrix after preprocessing
  SparseMatrix<double> LPreinfNotMerged;
  cotmatrix(VPreinf, F, LPreinfNotMerged);

  // Create identity matrix for smoothing
  SparseMatrix<double> I(V.rows(), V.rows());

  // Merge vertices (handle equality constraints)
  // In original implementation, boundary vertices of front and back surfaces should be merged (same Z coordinate)
  // Here we simplify: create empty merge map (can be added later if needed)
  unordered_map<int, set<int>> verticesToMerge;
  vector<pair<int, int>> bnd2bnd;
  
  // Note: Original implementation has more complex boundary correspondence handling
  // Here we simplify and don't handle front/back surface boundary merging for now
  // For complete implementation, need to find correspondence between front and back boundary vertices

  // Perform vertex merging
  vector<bool> removeV;
  vector<int> reindex;
  vector<vector<int>> verticesOfParts;
  
  MeshBuilder::meshMergeAndRemoveDuplicates(V, F, verticesToMerge, outputVertices,
                                            outputFaces, mb.parts1, verticesOfParts,
                                            removeV, reindex);

  // Update Z coordinates from VPreinf to output vertices
  MatrixXd VPreinfOut = outputVertices;
  for (int i = 0; i < V.rows(); i++) {
    if (!removeV[i]) {
      VPreinfOut(reindex[i], Z_COORD) = VPreinf(i, Z_COORD);
    }
  }

  outputVertices = VPreinfOut;

  return true;
}

