#include "histograma.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <mpi.h>

// AUXILIAR: Calcular índice del bin para un valor dado
// Devuelve -1 si está fuera de rango (aunque asumiremos datos válidos)
int get_bin_index(int value, int min_val, int bin_width, int num_bins)
{
  int idx = (value - min_val) / bin_width;
  if (idx >= num_bins)
    idx = num_bins - 1; // Clampear al último bin si es necesario
  if (idx < 0)
    idx = 0;
  return idx;
}

// 1. VERSIÓN PUNTO A PUNTO
std::vector<int> histograma_punto_a_punto(int rank, int num_procesos,
                                          const std::vector<int> &datos,
                                          int num_bins, int min_val, int max_val)
{
  int range = max_val - min_val + 1;
  int bin_width = std::ceil((double)range / num_bins);

  int n = 0;
  if (rank == 0)
    n = static_cast<int>(datos.size());

  // Broadcast manual de N para que todos sepan el tamaño (o asumimos N fijo conocido)
  // Para seguir tu estilo estricto, calcularemos elementos locales asumiendo N divisible
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int elementos_locales = n / num_procesos;
  std::vector<int> datos_locales(elementos_locales);
  std::vector<int> bins_locales(num_bins, 0);

  // Distribución de datos
  if (rank == 0)
  {
    // Copiar datos locales al proceso 0
    for (int i = 0; i < elementos_locales; i++)
      datos_locales[i] = datos[i];

    // Enviar al resto
    for (int p = 1; p < num_procesos; p++)
    {
      MPI_Send(&datos[p * elementos_locales], elementos_locales, MPI_INT, p, 0, MPI_COMM_WORLD);
    }
  }
  else
  {
    MPI_Recv(datos_locales.data(), elementos_locales, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Calculo local
  for (int val : datos_locales)
  {
    int idx = get_bin_index(val, min_val, bin_width, num_bins);
    bins_locales[idx]++;
  }

  // Recolección de resultados
  std::vector<int> bins_globales;
  if (rank == 0)
  {
    bins_globales = bins_locales; // Empezamos con mis conteos
    std::vector<int> bins_recibidos(num_bins);

    // Recibir de todos y sumar
    for (int p = 1; p < num_procesos; p++)
    {
      MPI_Recv(bins_recibidos.data(), num_bins, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Sumar al global
      for (int b = 0; b < num_bins; ++b)
      {
        bins_globales[b] += bins_recibidos[b];
      }
    }
  }
  else
  {
    // Enviar mis conteos al root
    MPI_Send(bins_locales.data(),
             num_bins,
             MPI_INT,
             0,
             1, // Tag 1
             MPI_COMM_WORLD);
  }

  return rank == 0 ? std::move(bins_globales) : std::vector<int>();
}

// 2. VERSION COLECTIVA
std::vector<int> histograma_colectiva(int rank, int num_procesos,
                                      const std::vector<int> &datos,
                                      int num_bins, int min_val, int max_val)
{
  int range = max_val - min_val + 1;
  int bin_width = std::ceil((double)range / num_bins);

  int n = 0;
  if (rank == 0)
    n = datos.size();
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int elementos_locales = n / num_procesos;
  std::vector<int> datos_locales(elementos_locales);
  std::vector<int> bins_locales(num_bins, 0);
  std::vector<int> bins_globales(rank == 0 ? num_bins : 0);

  // Distribución de datos
  MPI_Scatter(rank == 0 ? datos.data() : nullptr,
              elementos_locales,
              MPI_INT,
              datos_locales.data(),
              elementos_locales,
              MPI_INT,
              0,
              MPI_COMM_WORLD);

  // Calculo local
  for (int val : datos_locales)
  {
    int idx = get_bin_index(val, min_val, bin_width, num_bins);
    bins_locales[idx]++;
  }

  // Reducción
  MPI_Reduce(bins_locales.data(),
             rank == 0 ? bins_globales.data() : nullptr,
             num_bins,
             MPI_INT,
             MPI_SUM,
             0,
             MPI_COMM_WORLD);

  return rank == 0 ? std::move(bins_globales) : std::vector<int>();
}