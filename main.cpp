#include <mpi.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <numeric>
#include <string>
#include <vector>
#include <cmath>
#include <random>

#include "blelloch.h"
#include "histograma.h"

// [SCAN]
#define N 8
// #define N 64
// #define N 256
// #define N 1024

// [HISTOGRAMA]
// CONFIGURACIÓN
#define M 100       // Número total de datos
#define NUM_BINS 10 // Cantidad de barras del histograma
#define MIN_VAL 0   // Valor mínimo de los datos
#define MAX_VAL 99  // Valor máximo de los datos

std::vector<int> generar_datos_histograma(int cantidad, int min_val, int max_val)
{
  if (cantidad <= 0)
  {
    return {};
  }

  std::vector<int> datos(cantidad);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(min_val, max_val);

  for (int &valor : datos)
  {
    valor = dist(gen);
  }

  return datos;
}

void imprimir_vector_original(const std::vector<int> &datos)
{
  constexpr std::size_t kMaxPreview = 30; // Máximo de elementos a mostrar
  if (datos.size() <= kMaxPreview)
  {
    fmt::println("[HISTOGRAMA] Vector original ({} elementos): {}", datos.size(), datos);
  }
  else
  {
    std::vector<int> preview(datos.begin(), datos.begin() + kMaxPreview);
    fmt::println("[HISTOGRAMA] Vector original ({} elementos, primeros {}): {}", datos.size(), kMaxPreview, preview);
  }
}

bool ejercicio_scan_blelloch(int rank, int num_procesos)
{
  std::vector<int> datos(N);
  std::iota(datos.begin(), datos.end(), 1);

  std::string razon_error;
  if ((N & (N - 1)) != 0)
  {
    razon_error = "N debe ser potencia de dos para Blelloch";
  }
  else if (N < num_procesos)
  {
    razon_error = "N debe ser mayor o igual al numero de procesos";
  }
  else if (N % num_procesos != 0)
  {
    razon_error = "N debe ser divisible por el numero de procesos";
  }

  if (!razon_error.empty())
  {
    if (rank == 0)
    {
      fmt::println("[SCAN] Configuracion invalida: {}. Se omite el ejercicio.", razon_error);
    }
    return false;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const double inicio_pap = MPI_Wtime();
  std::vector<int> resultado_p2p = blelloch_punto_a_punto(rank, num_procesos, datos);
  MPI_Barrier(MPI_COMM_WORLD);
  const double tiempo_pap = MPI_Wtime() - inicio_pap;

  MPI_Barrier(MPI_COMM_WORLD);
  const double inicio_colectiva = MPI_Wtime();
  std::vector<int> resultado_colectiva = blelloch_colectiva(rank, num_procesos, datos);
  MPI_Barrier(MPI_COMM_WORLD);
  const double tiempo_colectiva = MPI_Wtime() - inicio_colectiva;

  if (rank == 0)
  {
    fmt::println("[SCAN]");
    fmt::println("Resultado PaP: {}", resultado_p2p);
    fmt::println("Resultado Colectiva: {}", resultado_colectiva);
    fmt::println("Comparativa de tiempos (ms):");
    fmt::println("Método            Tiempo");
    fmt::println("------------------------------");
    fmt::println("Punto a punto     {:>12.6f}", tiempo_pap * 1000.0);
    fmt::println("Colectiva         {:>12.6f}", tiempo_colectiva * 1000.0);
  }

  return true;
}

bool ejercicio_histograma(int rank, int num_procesos)
{
  const int rango = MAX_VAL - MIN_VAL + 1;

  std::string razon_error;
  if (M % num_procesos != 0)
  {
    razon_error = "M debe ser divisible por el numero de procesos";
  }
  else if (NUM_BINS <= 0)
  {
    razon_error = "NUM_BINS debe ser positivo";
  }
  else if (MAX_VAL < MIN_VAL)
  {
    razon_error = "MAX_VAL debe ser mayor o igual a MIN_VAL";
  }
  else if (rango < NUM_BINS)
  {
    razon_error = "El rango de valores debe ser al menos NUM_BINS";
  }

  if (!razon_error.empty())
  {
    if (rank == 0)
    {
      fmt::println("[HISTOGRAMA] Configuracion invalida: {}. Se omite el ejercicio.", razon_error);
    }
    return false;
  }

  std::vector<int> datos = generar_datos_histograma(M, MIN_VAL, MAX_VAL);
  if (rank == 0)
  {
    imprimir_vector_original(datos);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const double inicio_pap = MPI_Wtime();
  std::vector<int> bins_p2p = histograma_punto_a_punto(rank, num_procesos, datos, NUM_BINS, MIN_VAL, MAX_VAL);
  MPI_Barrier(MPI_COMM_WORLD);
  const double tiempo_pap = MPI_Wtime() - inicio_pap;

  MPI_Barrier(MPI_COMM_WORLD);
  const double inicio_colectiva = MPI_Wtime();
  std::vector<int> bins_colectiva = histograma_colectiva(rank, num_procesos, datos, NUM_BINS, MIN_VAL, MAX_VAL);
  MPI_Barrier(MPI_COMM_WORLD);
  const double tiempo_colectiva = MPI_Wtime() - inicio_colectiva;

  if (rank == 0)
  {
    fmt::println("[HISTOGRAMA]");
    fmt::println("\n Resultado PaP: {}", bins_p2p);
    fmt::println(" Resultado Colectiva: {}", bins_colectiva);
    fmt::println(" Comparativa de tiempos (ms):");
    fmt::println("Método            Tiempo");
    fmt::println("------------------------------");
    fmt::println("Punto a punto     {:>12.6f}", tiempo_pap * 1000.0);
    fmt::println("Colectiva         {:>12.6f}", tiempo_colectiva * 1000.0);
  }

  return true;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank, num_procesos;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procesos);
  if (rank == 0)
  {
    fmt::println("MPI Iniciado. N={} Procesos={}", N, num_procesos);
  }

  const bool scan_ok = ejercicio_scan_blelloch(rank, num_procesos);
  const bool hist_ok = ejercicio_histograma(rank, num_procesos);

  MPI_Finalize();
  return (scan_ok && hist_ok) ? 0 : 1;
}