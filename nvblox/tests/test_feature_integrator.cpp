/*
Copyright 2022 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <gtest/gtest.h>

#include "nvblox/integrators/projective_appearance_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/primitives/primitives.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/sensors/image.h"
#include "nvblox/tests/gpu_image_routines.h"
#include "nvblox/tests/integrator_utils.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

FeatureImage createFeatureImageWithValueEqualToIndexPlusOffset(
    const int rows, const int cols, const float offset) {
  FeatureImage image(rows, cols, MemoryType::kHost);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      for (size_t c = 0; c < image(y, x).size(); ++c) {
        image(y, x)[c] = static_cast<float>(c) + offset;
      }
    }
  }
  return image;
}

FeatureImage createConstantFeatureImage(const int rows, const int cols,
                                        const float value) {
  FeatureImage image(rows, cols, MemoryType::kHost);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      for (size_t c = 0; c < image(y, x).size(); ++c) {
        image(y, x)[c] = value;
      }
    }
  }
  return image;
}

class FeatureIntegratorTest : public ::testing::Test {
 protected:
  FeatureIntegratorTest() {
    // Create a scene with a sphere.
    scene_.aabb() = AxisAlignedBoundingBox(Vector3f(-5.0f, -5.0f, -5.0f),
                                           Vector3f(10.0f, 15.0f, 5.0f));
    const Eigen::Vector3f center = Vector3f(0.0f, 0.0f, 5.0f);
    scene_.addPrimitive(
        std::make_unique<primitives::Sphere>(center, kSphereRadius));
    scene_.generateLayerFromScene(kTruncationDistanceM, &tsdf_layer_);

    EXPECT_GT(tsdf_layer_.numBlocks(), 0);

    feature_integrator_.truncation_distance_vox(kTruncationDistanceVox);
    feature_integrator_.weighting_function_type(
        WeightingFunctionType::kConstantWeight);
  }

  // Integrate an image set to a constant value and check that all features in
  // the layer obtains this value
  void testConstantImage(float value) {
    FeatureImage image = createConstantFeatureImage(kHeight, kWidth, value);

    feature_integrator_.integrateFrame(
        MaskedFeatureImageConstView(image, kMaskActiveEverywhere),
        Transform::Identity(), camera_, tsdf_layer_, &feature_layer_,
        &updated_blocks_);
    // Check that voxel values equal the average between the two
    int num_active_voxels = 0;
    auto check_voxel_lambda = [&num_active_voxels, &value](
                                  const Index3D&, const Index3D&,
                                  const FeatureVoxel* voxel) -> void {
      if (voxel->weight > __float2half(0.F)) {
        ++num_active_voxels;
        for (const auto& item : voxel->feature) {
          const float item_as_float = __half2float(item);
          if (isinf(item_as_float)) {
            ASSERT_EQ(item_as_float, value);
          } else {
            // Use dynamic abs error threshold to support large ranges of input
            // values
            const float eps = fabs(item_as_float) / 1E6F;
            ASSERT_NEAR(item_as_float, value, eps);
          }
        }
      }
    };
    callFunctionOnAllVoxels<FeatureVoxel>(feature_layer_, check_voxel_lambda);
    EXPECT_GT(num_active_voxels, 0);
  }

  // Truncation distance
  constexpr static float kVoxelSize = 0.2F;
  constexpr static float kTruncationDistanceVox = 2;
  constexpr static float kTruncationDistanceM =
      kTruncationDistanceVox * kVoxelSize;

  constexpr static float kSphereRadius = 2.0f;
  primitives::Scene scene_;
  TsdfLayer tsdf_layer_{kVoxelSize, MemoryType::kHost};

  // Test camera (small image to improve test speed)
  constexpr static float kFu = 45;
  constexpr static float kFv = 45;
  constexpr static int kWidth = 64;
  constexpr static int kHeight = 48;
  constexpr static float kCu = static_cast<float>(kWidth) / 2.0f;
  constexpr static float kCv = static_cast<float>(kHeight) / 2.0f;
  Camera camera_{kFu, kFv, kCu, kCv, kWidth, kHeight};

  ProjectiveFeatureIntegrator feature_integrator_;
  FeatureLayer feature_layer_{kVoxelSize, MemoryType::kHost};

  std::vector<Index3D> updated_blocks_;
};

TEST_F(FeatureIntegratorTest, IntegrateSingleFeatureImage) {
  FeatureImage image =
      createFeatureImageWithValueEqualToIndexPlusOffset(kHeight, kWidth, 0);

  feature_integrator_.integrateFrame(
      MaskedFeatureImageConstView(image, kMaskActiveEverywhere),
      Transform::Identity(), camera_, tsdf_layer_, &feature_layer_,
      &updated_blocks_);

  // Check that all voxels are obtained from features of the input image
  int num_active_voxels = 0;
  auto check_voxel_lambda = [&num_active_voxels](
                                const Index3D&, const Index3D&,
                                const FeatureVoxel* voxel) -> void {
    if (voxel->weight > __float2half(0.F)) {
      ++num_active_voxels;
      int expected = 0;
      EXPECT_GT(voxel->feature.size(), 0);
      for (const auto& item : voxel->feature) {
        EXPECT_EQ(__half2float(item), expected++);
      }
    }
  };
  callFunctionOnAllVoxels<FeatureVoxel>(feature_layer_, check_voxel_lambda);

  // Sanitiy check
  EXPECT_GT(num_active_voxels, 0);
}

TEST_F(FeatureIntegratorTest, IntegrateTwoFeatureImages) {
  // Integrate two feature images where the second has its values offset
  // by one
  FeatureImage image0 =
      createFeatureImageWithValueEqualToIndexPlusOffset(kHeight, kWidth, 0);
  FeatureImage image1 =
      createFeatureImageWithValueEqualToIndexPlusOffset(kHeight, kWidth, 1);

  // Integrate both of them
  feature_integrator_.integrateFrame(
      MaskedFeatureImageConstView(image0, kMaskActiveEverywhere),
      Transform::Identity(), camera_, tsdf_layer_, &feature_layer_,
      &updated_blocks_);
  feature_integrator_.integrateFrame(
      MaskedFeatureImageConstView(image1, kMaskActiveEverywhere),
      Transform::Identity(), camera_, tsdf_layer_, &feature_layer_,
      &updated_blocks_);
  // Check that voxel values equal the average between the two
  int num_active_voxels = 0;
  auto check_voxel_lambda = [&num_active_voxels](
                                const Index3D&, const Index3D&,
                                const FeatureVoxel* voxel) -> void {
    if (voxel->weight > __float2half(0.F)) {
      ++num_active_voxels;
      int expected = 0;
      for (const auto& item : voxel->feature) {
        // We compare relative error since float16 used for feature elements
        // becomes inaccurate for large values, e.g. 1024.0 + 0.5 = 1024.0
        const float relative_error =
            std::abs(1.0F - __half2float(item) / (expected + 0.5));

        EXPECT_LE(relative_error, 1E-3);
        ++expected;
      }
    }
  };
  callFunctionOnAllVoxels<FeatureVoxel>(feature_layer_, check_voxel_lambda);
  EXPECT_GT(num_active_voxels, 0);
}

TEST_F(FeatureIntegratorTest, CornerCaseZeroFlt) { testConstantImage(0.F); }

TEST_F(FeatureIntegratorTest, CornerCaseMaxFlt) {
  testConstantImage(std::numeric_limits<__half>::max());
}

TEST_F(FeatureIntegratorTest, CornerCaseMinFlt) {
  testConstantImage(std::numeric_limits<__half>::min());
}

TEST_F(FeatureIntegratorTest, CornerCaseLowestFlt) {
  testConstantImage(std::numeric_limits<__half>::lowest());
}

TEST_F(FeatureIntegratorTest, CornerCaseInfinityFlt) {
  testConstantImage(std::numeric_limits<__half>::infinity());
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
