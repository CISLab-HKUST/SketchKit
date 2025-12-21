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

#ifndef SKETCH2MODEL_H
#define SKETCH2MODEL_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>
#include <string>

#include "image/image.h"

/**
 * @brief Convert 2D sketch to 3D model (following MonsterMash original implementation)
 * 
 * @param outlineImgs List of outline images (white background, black outline)
 * @param regionImgs List of region filled images (black background, white/colored regions)
 * @param outputVertices Output 3D vertex coordinates (N x 3 matrix, N vertices)
 * @param outputFaces Output face indices (M x 3 matrix, M triangles)
 * @param triangleOpts Triangle library options (default: "pqa25QYY")
 * @param subsFactor Subsampling factor (default: 2)
 * @param smoothFactor Smoothing factor (default: 10)
 * @param defaultInflationAmount Default inflation amount (default: 2.0)
 * @param regionInflationAmounts Inflation amount for each region (optional, uses default if empty)
 * 
 * @return true on success, false on failure
 * 
 * Implementation notes:
 * 1. Convert 2D outlines and regions to planar meshes (triangulation)
 * 2. Merge meshes from multiple regions
 * 3. Implement 3D inflation via optimization problem (compute Z coordinates)
 * 4. Handle boundary conditions and vertex merging
 * 5. Output final 3D mesh
 */
bool sketch2model(
    const std::vector<Imguc> &outlineImgs,
    const std::vector<Imguc> &regionImgs,
    Eigen::MatrixXd &outputVertices,
    Eigen::MatrixXi &outputFaces,
    const std::string &triangleOpts = "pqa25QYY",
    const int subsFactor = 2,
    const int smoothFactor = 10,
    const double defaultInflationAmount = 2.0,
    const std::vector<double> &regionInflationAmounts = std::vector<double>());

#endif  // SKETCH2MODEL_H

