#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// Helper: Trilinear interpolation on 3D grid
inline void trilinear_interp(const std::vector<double>& src_lut, int src_N,
                             double r, double g, double b,
                             double& out_r, double& out_g, double& out_b) {
  // Clamp input coordinates to [0, 1]
  r = std::min(std::max(r, 0.0), 1.0);
  g = std::min(std::max(g, 0.0), 1.0);
  b = std::min(std::max(b, 0.0), 1.0);

  double r_scaled = r * (src_N - 1);
  double g_scaled = g * (src_N - 1);
  double b_scaled = b * (src_N - 1);

  int r0 = static_cast<int>(std::floor(r_scaled));
  int g0 = static_cast<int>(std::floor(g_scaled));
  int b0 = static_cast<int>(std::floor(b_scaled));

  int r1 = std::min(r0 + 1, src_N - 1);
  int g1 = std::min(g0 + 1, src_N - 1);
  int b1 = std::min(b0 + 1, src_N - 1);

  double dr = r_scaled - r0;
  double dg = g_scaled - g0;
  double db = b_scaled - b0;

  // Fetch 8 corners of bounding voxel
  auto get_val = [&](int cr, int cg, int cb, int ch) {
    int idx = (cr + src_N * (cg + src_N * cb)) * 3 + ch;
    return src_lut[idx];
  };

  for (int ch = 0; ch < 3; ++ch) {
    double c000 = get_val(r0, g0, b0, ch);
    double c100 = get_val(r1, g0, b0, ch);
    double c010 = get_val(r0, g1, b0, ch);
    double c110 = get_val(r1, g1, b0, ch);
    double c001 = get_val(r0, g0, b1, ch);
    double c101 = get_val(r1, g0, b1, ch);
    double c011 = get_val(r0, g1, b1, ch);
    double c111 = get_val(r1, g1, b1, ch);

    double c00 = c000 * (1.0 - dr) + c100 * dr;
    double c10 = c010 * (1.0 - dr) + c110 * dr;
    double c01 = c001 * (1.0 - dr) + c101 * dr;
    double c11 = c011 * (1.0 - dr) + c111 * dr;

    double c0 = c00 * (1.0 - dg) + c10 * dg;
    double c1 = c01 * (1.0 - dg) + c11 * dg;

    double final_val = c0 * (1.0 - db) + c1 * db;

    if (ch == 0) out_r = final_val;
    else if (ch == 1) out_g = final_val;
    else out_b = final_val;
  }
}

// [[Rcpp::export]]
List convert_resample_lut_cpp(NumericMatrix input_lut,
                              int src_N,
                              int target_N,
                              std::string input_format = "cube",
                              bool force_0 = false,
                              bool force_1 = false) {
  int src_cube_size = src_N * src_N * src_N;
  if (input_lut.nrow() != src_cube_size || input_lut.ncol() != 3) {
    stop("Dimensions of input_lut do not match src_N^3 x 3.");
  }

  // 1. Convert input matrix into canonical 3D grid layout [r + N*(g + N*b)]
  std::vector<double> src_grid(src_cube_size * 3);

  if (input_format == "cube") {
    #pragma omp parallel for
    for (int i = 0; i < src_cube_size; ++i) {
      src_grid[i * 3]     = input_lut(i, 0);
      src_grid[i * 3 + 1] = input_lut(i, 1);
      src_grid[i * 3 + 2] = input_lut(i, 2);
    }
  } else if (input_format == "hald") {
    // Hald layout: red varies fastest, then green, then blue
    // HaldCLUT spatial layout mapping (Level = sqrt(src_N))
    int level = static_cast<int>(std::round(std::sqrt(static_cast<double>(src_N))));
    if (level * level != src_N) {
      stop("For HaldCLUT input, src_N must be a perfect square (e.g. 8 for Level 8 -> N=64).");
    }

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < src_N; ++b) {
      for (int g = 0; g < src_N; ++g) {
        for (int r = 0; r < src_N; ++r) {
          int canonical_idx = (r + src_N * (g + src_N * b)) * 3;

          // Compute HaldCLUT 2D pixel index (Standard HaldCLUT ordering)
          int hald_x = (r % level) + (g % level) * level + (b % level) * level * level; // fast axis
          int hald_y = (r / level) + (g / level) * level + (b / level) * level * level; // slow axis
          int hald_pixel_idx = hald_y * (src_N * level) + hald_x; 
          
          // Fallback for linear-flattened Hald matrix
          if (hald_pixel_idx >= src_cube_size) hald_pixel_idx = r + src_N * (g + src_N * b);

          src_grid[canonical_idx]     = input_lut(hald_pixel_idx, 0);
          src_grid[canonical_idx + 1] = input_lut(hald_pixel_idx, 1);
          src_grid[canonical_idx + 2] = input_lut(hald_pixel_idx, 2);
        }
      }
    }
  } else {
    stop("Unsupported input_format. Use 'cube' or 'hald'.");
  }

  // 2. Resample onto target grid (N_target)
  int target_cube_size = target_N * target_N * target_N;
  std::vector<double> target_grid(target_cube_size * 3);

  #pragma omp parallel for collapse(3)
  for (int b = 0; b < target_N; ++b) {
    for (int g = 0; g < target_N; ++g) {
      for (int r = 0; r < target_N; ++r) {
        int target_idx = (r + target_N * (g + target_N * b)) * 3;

        double norm_r = static_cast<double>(r) / (target_N - 1);
        double norm_g = static_cast<double>(g) / (target_N - 1);
        double norm_b = static_cast<double>(b) / (target_N - 1);

        if (src_N == target_N) {
          target_grid[target_idx]     = src_grid[target_idx];
          target_grid[target_idx + 1] = src_grid[target_idx + 1];
          target_grid[target_idx + 2] = src_grid[target_idx + 2];
        } else {
          double out_r, out_g, out_b;
          trilinear_interp(src_grid, src_N, norm_r, norm_g, norm_b, out_r, out_g, out_b);
          target_grid[target_idx]     = std::min(std::max(out_r, 0.0), 1.0);
          target_grid[target_idx + 1] = std::min(std::max(out_g, 0.0), 1.0);
          target_grid[target_idx + 2] = std::min(std::max(out_b, 0.0), 1.0);
        }
      }
    }
  }

  // 3. Optional forced black (0,0,0) and white (1,1,1) preservation
  if (force_0) {
    target_grid[0] = 0.0;
    target_grid[1] = 0.0;
    target_grid[2] = 0.0;
  }
  if (force_1) {
    int last_idx = (target_cube_size - 1) * 3;
    target_grid[last_idx]     = 1.0;
    target_grid[last_idx + 1] = 1.0;
    target_grid[last_idx + 2] = 1.0;
  }

  // 4. Build output matrices
  NumericMatrix cube_out(target_cube_size, 3);
  #pragma omp parallel for
  for (int i = 0; i < target_cube_size; ++i) {
    cube_out(i, 0) = target_grid[i * 3];
    cube_out(i, 1) = target_grid[i * 3 + 1];
    cube_out(i, 2) = target_grid[i * 3 + 2];
  }

  // 5. Build Hald Matrix if target_N is a perfect square
  int target_level = static_cast<int>(std::round(std::sqrt(static_cast<double>(target_N))));
  bool can_make_hald = (target_level * target_level == target_N);

  if (can_make_hald) {
    NumericMatrix hald_out(target_cube_size, 3);

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < target_N; ++b) {
      for (int g = 0; g < target_N; ++g) {
        for (int r = 0; r < target_N; ++r) {
          int canonical_idx = (r + target_N * (g + target_N * b)) * 3;

          int hald_x = (r % target_level) + (g % target_level) * target_level + (b % target_level) * target_level * target_level;
          int hald_y = (r / target_level) + (g / target_level) * target_level + (b / target_level) * target_level * target_level;
          int hald_pixel_idx = hald_y * (target_N * target_level) + hald_x;

          if (hald_pixel_idx < target_cube_size) {
            hald_out(hald_pixel_idx, 0) = target_grid[canonical_idx];
            hald_out(hald_pixel_idx, 1) = target_grid[canonical_idx + 1];
            hald_out(hald_pixel_idx, 2) = target_grid[canonical_idx + 2];
          }
        }
      }
    }

    return List::create(
      Named("cube") = cube_out,
      Named("hald") = hald_out,
      Named("hald_dim") = Dimension(target_N * target_level, target_N * target_level)
    );
  }

  return List::create(
    Named("cube") = cube_out,
    Named("hald") = R_NilValue,
    Named("hald_dim") = R_NilValue
  );
}
