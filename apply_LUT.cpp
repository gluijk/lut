#include <Rcpp.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// Helper: Trilinear interpolation inside the 3D LUT
inline void lookup_trilinear(const std::vector<double>& lut, int N, double r, double g, double b, double& out_r, double& out_g, double& out_b) {
  // Scale RGB values [0, 1] to LUT grid coordinates [0, N - 1]
  double r_grid = r * (N - 1);
  double g_grid = g * (N - 1);
  double b_grid = b * (N - 1);

  int r0 = std::min(static_cast<int>(std::floor(r_grid)), N - 2);
  int g0 = std::min(static_cast<int>(std::floor(g_grid)), N - 2);
  int b0 = std::min(static_cast<int>(std::floor(b_grid)), N - 2);

  int r1 = r0 + 1;
  int g1 = g0 + 1;
  int b1 = b0 + 1;

  double dr = r_grid - r0;
  double dg = g_grid - g0;
  double db = b_grid - b0;

  // Lambda to fetch flat index
  auto get_rgb = [&](int r_idx, int g_idx, int b_idx, int channel) {
    int idx = (r_idx + N * (g_idx + N * b_idx)) * 3 + channel;
    return lut[idx];
  };

  // Interpolate each channel
  double res[3] = {0.0, 0.0, 0.0};
  for (int c = 0; c < 3; ++c) {
    double c000 = get_rgb(r0, g0, b0, c);
    double c100 = get_rgb(r1, g0, b0, c);
    double c010 = get_rgb(r0, g1, b0, c);
    double c110 = get_rgb(r1, g1, b0, c);
    double c001 = get_rgb(r0, g0, b1, c);
    double c101 = get_rgb(r1, g0, b1, c);
    double c011 = get_rgb(r0, g1, b1, c);
    double c111 = get_rgb(r1, g1, b1, c);

    double c00 = c000 * (1.0 - dr) + c100 * dr;
    double c10 = c010 * (1.0 - dr) + c110 * dr;
    double c01 = c001 * (1.0 - dr) + c101 * dr;
    double c11 = c011 * (1.0 - dr) + c111 * dr;

    double c0 = c00 * (1.0 - dg) + c10 * dg;
    double c1 = c01 * (1.0 - dg) + c11 * dg;

    res[c] = c0 * (1.0 - db) + c1 * db;
  }

  out_r = std::min(std::max(res[0], 0.0), 1.0);
  out_g = std::min(std::max(res[1], 0.0), 1.0);
  out_b = std::min(std::max(res[2], 0.0), 1.0);
}

// [[Rcpp::export]]
NumericMatrix apply_lut_cube_cpp(NumericMatrix img_mat, std::string cube_path) {
  std::ifstream infile(cube_path);
  if (!infile.is_open()) {
    stop("Could not open .cube file.");
  }

  std::string line;
  int N = 0;
  std::vector<double> lut;

  while (std::getline(infile, line)) {
    if (line.empty() || line[0] == '#') continue;

    if (line.rfind("LUT_3D_SIZE", 0) == 0) {
      std::stringstream ss(line);
      std::string token;
      ss >> token >> N;
      lut.reserve(N * N * N * 3);
      continue;
    }

    std::stringstream ss(line);
    double r, g, b;
    if (ss >> r >> g >> b) {
      lut.push_back(r);
      lut.push_back(g);
      lut.push_back(b);
    }
  }

  if (N == 0 || lut.size() != static_cast<size_t>(N * N * N * 3)) {
    stop("Invalid .cube file formatting or missing LUT_3D_SIZE.");
  }

  int n_pixels = img_mat.nrow();
  NumericMatrix out_mat(n_pixels, 3);

  #pragma omp parallel for
  for (int i = 0; i < n_pixels; ++i) {
    double r_in = img_mat(i, 0);
    double g_in = img_mat(i, 1);
    double b_in = img_mat(i, 2);

    double r_out, g_out, b_out;
    lookup_trilinear(lut, N, r_in, g_in, b_in, r_out, g_out, b_out);

    out_mat(i, 0) = r_out;
    out_mat(i, 1) = g_out;
    out_mat(i, 2) = b_out;
  }

  return out_mat;
}

// [[Rcpp::export]]
NumericMatrix apply_lut_array_cpp(NumericMatrix img_mat, NumericVector lut_vector, int N) {
  std::vector<double> lut = as<std::vector<double>>(lut_vector);
  int n_pixels = img_mat.nrow();
  NumericMatrix out_mat(n_pixels, 3);

  #pragma omp parallel for
  for (int i = 0; i < n_pixels; ++i) {
    double r_in = img_mat(i, 0);
    double g_in = img_mat(i, 1);
    double b_in = img_mat(i, 2);

    double r_out, g_out, b_out;
    lookup_trilinear(lut, N, r_in, g_in, b_in, r_out, g_out, b_out);

    out_mat(i, 0) = r_out;
    out_mat(i, 1) = g_out;
    out_mat(i, 2) = b_out;
  }

  return out_mat;
}
