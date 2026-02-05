#include <mpi.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <numeric>
#include <string>
#include <vector>
#include <cmath>

#include "blelloch.h"

// #define N 8
// #define N 64
// #define N 256
#define N 1024

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank, num_procesos;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procesos);

  std::vector<int> datos(N);
  for (int i = 0; i < N; ++i)
    datos[i] = i + 1; // {1,2,3,4,...,N}

  // Validaciones
  std::string razon_error;
  if ((N & (N - 1)) != 0)
  {
    razon_error = "N debe ser potencia de dos para Blelloch";
  }
  else if (N % num_procesos != 0)
  {
    razon_error = "N debe ser divisible por el numero de procesos";
  }

  if (rank == 0)
  {
    fmt::print("MPI Iniciado. N={} Procesos={}\n", N, num_procesos);
    if (!razon_error.empty())
    {
      fmt::print("Configuracion invalida: {}. Abortando.\n", razon_error);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  if (!razon_error.empty())
  {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Punto a Punto
  MPI_Barrier(MPI_COMM_WORLD);
  const double inicio_pap = MPI_Wtime();
  std::vector<int> resultado_p2p = blelloch_punto_a_punto(rank, num_procesos, datos);
  MPI_Barrier(MPI_COMM_WORLD);
  const double tiempo_pap = MPI_Wtime() - inicio_pap;

  // Colectiva
  MPI_Barrier(MPI_COMM_WORLD);
  const double inicio_colectiva = MPI_Wtime();
  std::vector<int> resultado_colectiva = blelloch_colectiva(rank, num_procesos, datos);
  MPI_Barrier(MPI_COMM_WORLD);
  const double tiempo_colectiva = MPI_Wtime() - inicio_colectiva;

  if (rank == 0)
  {
    fmt::println("Resultado PaP:       {}\n", resultado_p2p);
    fmt::println("Resultado Colectiva: {}\n", resultado_colectiva);

    fmt::println("\nComparativa de tiempos (ms):");
    fmt::println("Metodo             Tiempo");
    fmt::println("----------------------------");
    fmt::println("Punto a punto      {:.6f}", tiempo_pap * 1000);
    fmt::println("Colectiva          {:.6f}", tiempo_colectiva * 1000);
  }

  MPI_Finalize();
  return 0;
}