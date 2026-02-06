#ifndef HISTOGRAMA_H
#define HISTOGRAMA_H

#include <vector>

int get_bin_index(int value, int min_val, int bin_width, int num_bins);

std::vector<int> histograma_punto_a_punto(int rank, int num_procesos,
                                          const std::vector<int> &datos,
                                          int num_bins, int min_val, int max_val);
std::vector<int> histograma_colectiva(int rank, int num_procesos,
                                      const std::vector<int> &datos,
                                      int num_bins, int min_val, int max_val);

#endif // HISTOGRAMA_H