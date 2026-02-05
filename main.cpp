#include <mpi.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <numeric>
#include <string>
#include <vector>
#include <cmath>

#include "blelloch.h"

#define N 64

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
  const double inicio_p2p = MPI_Wtime();
  std::vector<int> resultado_p2p = blelloch_punto_a_punto(rank, num_procesos, datos);
  MPI_Barrier(MPI_COMM_WORLD);
  const double tiempo_p2p = MPI_Wtime() - inicio_p2p;

  // Colectiva
  MPI_Barrier(MPI_COMM_WORLD);
  const double inicio_colectiva = MPI_Wtime();
  std::vector<int> resultado_colectiva = blelloch_colectiva(rank, num_procesos, datos);
  MPI_Barrier(MPI_COMM_WORLD);
  const double tiempo_colectiva = MPI_Wtime() - inicio_colectiva;

  if (rank == 0)
  {
    fmt::print("Resultado P2P:       {}\n", resultado_p2p);
    fmt::print("Resultado Colectiva: {}\n", resultado_colectiva);

    fmt::println("\nComparativa de tiempos (s):");
    fmt::println("Metodo             Tiempo");
    fmt::println("----------------------------");
    fmt::println("Punto a punto      {:.6f}", tiempo_p2p);
    fmt::println("Colectiva          {:.6f}", tiempo_colectiva);
  }

  MPI_Finalize();
  return 0;
}