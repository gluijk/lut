#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

struct KnownNode {
  double r_in, g_in, b_in;
  double r_out, g_out, b_out;
};

// [[Rcpp::export]]
NumericVector build_lut3d_cpp(NumericMatrix img_in, 
                              NumericMatrix img_out, 
                              int N, 
                              int k_neighbors = 8, 
                              double power = 2.0, 
                              bool apply_smooth = false, 
                              double sigma = 0.8,
                              bool force_0 = false,
                              bool force_1 = false) {
  int n_pixels = img_in.nrow();
  int cube_size = N * N * N;
  
  std::vector<double> sum_r(cube_size, 0.0);
  std::vector<double> sum_g(cube_size, 0.0);
  std::vector<double> sum_b(cube_size, 0.0);
  std::vector<int> counts(cube_size, 0);
  
  // 1. Acumulación paralela mediante cuantización al nodo más cercano
  #pragma omp parallel
  {
    std::vector<double> local_sum_r(cube_size, 0.0);
    std::vector<double> local_sum_g(cube_size, 0.0);
    std::vector<double> local_sum_b(cube_size, 0.0);
    std::vector<int> local_counts(cube_size, 0);
    
    #pragma omp for nowait
    for (int i = 0; i < n_pixels; ++i) {
      int r_idx = std::min(std::max(static_cast<int>(std::round(img_in(i, 0) * (N - 1))), 0), N - 1);
      int g_idx = std::min(std::max(static_cast<int>(std::round(img_in(i, 1) * (N - 1))), 0), N - 1);
      int b_idx = std::min(std::max(static_cast<int>(std::round(img_in(i, 2) * (N - 1))), 0), N - 1);
      
      int cell_idx = r_idx + N * (g_idx + N * b_idx);
      
      local_sum_r[cell_idx] += img_out(i, 0);
      local_sum_g[cell_idx] += img_out(i, 1);
      local_sum_b[cell_idx] += img_out(i, 2);
      local_counts[cell_idx]++;
    }
    
    #pragma omp critical
    {
      for (int c = 0; c < cube_size; ++c) {
        sum_r[c] += local_sum_r[c];
        sum_g[c] += local_sum_g[c];
        sum_b[c] += local_sum_b[c];
        counts[c] += local_counts[c];
      }
    }
  }
  
  // 2. Recopilar nodos observados
  std::vector<KnownNode> known_nodes;
  known_nodes.reserve(cube_size);
  
  for (int b = 0; b < N; ++b) {
    for (int g = 0; g < N; ++g) {
      for (int r = 0; r < N; ++r) {
        int cell_idx = r + N * (g + N * b);
        if (counts[cell_idx] > 0) {
          KnownNode kn;
          kn.r_in = static_cast<double>(r) / (N - 1);
          kn.g_in = static_cast<double>(g) / (N - 1);
          kn.b_in = static_cast<double>(b) / (N - 1);
          
          kn.r_out = sum_r[cell_idx] / counts[cell_idx];
          kn.g_out = sum_g[cell_idx] / counts[cell_idx];
          kn.b_out = sum_b[cell_idx] / counts[cell_idx];
          
          known_nodes.push_back(kn);
        }
      }
    }
  }
  
  std::vector<double> raw_lut(cube_size * 3);
  int num_known = known_nodes.size();
  
  // 3. Generación del cubo e interpolación IDW
  #pragma omp parallel for collapse(3)
  for (int b = 0; b < N; ++b) {
    for (int g = 0; g < N; ++g) {
      for (int r = 0; r < N; ++r) {
        int cell_idx = r + N * (g + N * b);
        int lut_idx = cell_idx * 3;
        
        if (counts[cell_idx] > 0) {
          raw_lut[lut_idx]     = sum_r[cell_idx] / counts[cell_idx];
          raw_lut[lut_idx + 1] = sum_g[cell_idx] / counts[cell_idx];
          raw_lut[lut_idx + 2] = sum_b[cell_idx] / counts[cell_idx];
        } else if (num_known > 0) {
          double curr_r = static_cast<double>(r) / (N - 1);
          double curr_g = static_cast<double>(g) / (N - 1);
          double curr_b = static_cast<double>(b) / (N - 1);
          
          std::vector<std::pair<double, int>> dists(num_known);
          for (int k = 0; k < num_known; ++k) {
            double dr = curr_r - known_nodes[k].r_in;
            double dg = curr_g - known_nodes[k].g_in;
            double db = curr_b - known_nodes[k].b_in;
            dists[k] = {dr * dr + dg * dg + db * db, k};
          }
          
          int effective_k = std::min(k_neighbors, num_known);
          std::partial_sort(dists.begin(), dists.begin() + effective_k, dists.end());
          
          double weight_sum = 0.0;
          double interp_r = 0.0, interp_g = 0.0, interp_b = 0.0;
          
          for (int k = 0; k < effective_k; ++k) {
            double dist = std::sqrt(dists[k].first);
            if (dist < 1e-8) dist = 1e-8;
            double w = 1.0 / std::pow(dist, power);
            int idx = dists[k].second;
            
            interp_r += w * known_nodes[idx].r_out;
            interp_g += w * known_nodes[idx].g_out;
            interp_b += w * known_nodes[idx].b_out;
            weight_sum += w;
          }
          
          raw_lut[lut_idx]     = std::min(std::max(interp_r / weight_sum, 0.0), 1.0);
          raw_lut[lut_idx + 1] = std::min(std::max(interp_g / weight_sum, 0.0), 1.0);
          raw_lut[lut_idx + 2] = std::min(std::max(interp_b / weight_sum, 0.0), 1.0);
        } else {
          raw_lut[lut_idx]     = static_cast<double>(r) / (N - 1);
          raw_lut[lut_idx + 1] = static_cast<double>(g) / (N - 1);
          raw_lut[lut_idx + 2] = static_cast<double>(b) / (N - 1);
        }
      }
    }
  }

  // 4. Suavizado Gaussiano 3D opcional
  NumericVector final_lut(cube_size * 3);
  if (!apply_smooth) {
    std::copy(raw_lut.begin(), raw_lut.end(), final_lut.begin());
  } else {
    int radius = static_cast<int>(std::ceil(2.0 * sigma));
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < N; ++b) {
      for (int g = 0; g < N; ++g) {
        for (int r = 0; r < N; ++r) {
          int cell_idx = r + N * (g + N * b);
          int lut_idx = cell_idx * 3;

          double total_weight = 0.0;
          double sm_r = 0.0, sm_g = 0.0, sm_b = 0.0;

          for (int db = -radius; db <= radius; ++db) {
            int nb = b + db;
            if (nb < 0 || nb >= N) continue;
            for (int dg = -radius; dg <= radius; ++dg) {
              int ng = g + dg;
              if (ng < 0 || ng >= N) continue;
              for (int dr = -radius; dr <= radius; ++dr) {
                int nr = r + dr;
                if (nr < 0 || nr >= N) continue;

                double dist_sq = dr * dr + dg * dg + db * db;
                double w = std::exp(-dist_sq / (2.0 * sigma * sigma));

                int n_idx = (nr + N * (ng + N * nb)) * 3;
                sm_r += raw_lut[n_idx] * w;
                sm_g += raw_lut[n_idx + 1] * w;
                sm_b += raw_lut[n_idx + 2] * w;
                total_weight += w;
              }
            }
          }

          final_lut[lut_idx]     = sm_r / total_weight;
          final_lut[lut_idx + 1] = sm_g / total_weight;
          final_lut[lut_idx + 2] = sm_b / total_weight;
        }
      }
    }
  }

  // 5. Forzar extremos (preservación de puntos negros y blancos puros)
  if (force_0) {
    // Primer nodo: (0, 0, 0)
    final_lut[0] = 0.0;
    final_lut[1] = 0.0;
    final_lut[2] = 0.0;
  }

  if (force_1) {
    // Último nodo: (N-1, N-1, N-1)
    int last_idx = (cube_size - 1) * 3;
    final_lut[last_idx]     = 1.0;
    final_lut[last_idx + 1] = 1.0;
    final_lut[last_idx + 2] = 1.0;
  }

  return final_lut;
}