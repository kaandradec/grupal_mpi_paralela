#ifndef BLELLOCH_H
#define BLELLOCH_H

#include <vector>

void blellochSerial(int *v, int n);
std::vector<int> blelloch_punto_a_punto(int rank, int num_procesos, std::vector<int> datos);
std::vector<int> blelloch_colectiva(int rank, int num_procesos, std::vector<int> datos);

#endif // BLELLOCH_H
