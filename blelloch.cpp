#include "blelloch.h"
#include <cmath>
#include <vector>
#include <mpi.h>

// AUXILIAR: Scan Blelloch Serial
void blellochSerial(int *v, int n)
{
  const int levels = static_cast<int>(std::log2(n));

  // Up-Sweep
  for (int d = 0; d < levels; ++d)
  {
    const int stride = 1 << (d + 1);
    for (int i = 0; i < n; i += stride)
    {
      v[i + stride - 1] += v[i + (stride / 2) - 1];
    }
  }

  v[n - 1] = 0;

  // Down-Sweep
  for (int d = levels - 1; d >= 0; --d)
  {
    const int stride = 1 << (d + 1);
    for (int i = 0; i < n; i += stride)
    {
      const int left = i + (stride / 2) - 1;
      const int right = i + stride - 1;
      const int temp = v[left];
      v[left] = v[right];
      v[right] += temp;
    }
  }
}

// 1. VERSIÓN PUNTO A PUNTO
std::vector<int> blelloch_punto_a_punto(int rank, int num_procesos, std::vector<int> datos)
{
  const int n = static_cast<int>(datos.size());
  const int elementos_locales = n / num_procesos;

  std::vector<int> vector_global = std::move(datos);
  std::vector<int> vector_local(elementos_locales);

  // Distribuir datos
  if (rank == 0)
  {
    // Copiar parte local al proceso 0
    for (int i = 0; i < elementos_locales; i++)
      vector_local[i] = vector_global[i];

    // Enviar al resto
    for (int p = 1; p < num_procesos; p++)
    {
      MPI_Send(
          &vector_global[p * elementos_locales],
          elementos_locales,
          MPI_INT,
          p,
          0,
          MPI_COMM_WORLD);
    }
  }
  else
  {
    MPI_Recv(
        vector_local.data(),
        elementos_locales,
        MPI_INT,
        0,
        0,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);
  }

  // Calculo de suma local
  int suma_local = 0;
  for (int i = 0; i < elementos_locales; i++)
  {
    suma_local += vector_local[i];
  }

  // Calculo de offsets
  int offset_propio = 0;

  if (rank == 0)
  {
    std::vector<int> sumas_recibidas(num_procesos);
    std::vector<int> offsets_calculados(num_procesos);

    sumas_recibidas[0] = suma_local;

    // Recibir sumas de todos
    for (int p = 1; p < num_procesos; p++)
    {
      MPI_Recv(&sumas_recibidas[p], 1, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Calcular Offsets (Scan Serial de las sumas)
    int acumulado = 0;
    for (int i = 0; i < num_procesos; i++)
    {
      offsets_calculados[i] = acumulado;
      acumulado += sumas_recibidas[i];
    }

    offset_propio = offsets_calculados[0];

    // Enviar offsets de vuelta
    for (int p = 1; p < num_procesos; p++)
    {
      MPI_Send(&offsets_calculados[p], 1, MPI_INT, p, 2, MPI_COMM_WORLD);
    }
  }
  else
  {
    // Enviar mi suma
    MPI_Send(&suma_local, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    // Recibir mi offset
    MPI_Recv(&offset_propio, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Cálculo final
  blellochSerial(vector_local.data(), elementos_locales);

  for (int i = 0; i < elementos_locales; i++)
  {
    vector_local[i] += offset_propio;
  }

  // Recolección de resultados
  if (rank == 0)
  {
    // Copiar mi resultado
    for (int i = 0; i < elementos_locales; i++)
      vector_global[i] = vector_local[i];

    // Recibir del resto
    for (int p = 1; p < num_procesos; p++)
    {
      MPI_Recv(
          &vector_global[p * elementos_locales],
          elementos_locales,
          MPI_INT,
          p,
          3,
          MPI_COMM_WORLD,
          MPI_STATUS_IGNORE);
    }
  }
  else
  {
    MPI_Send(vector_local.data(), elementos_locales, MPI_INT, 0, 3, MPI_COMM_WORLD);
  }
  return rank == 0 ? std::move(vector_global) : std::vector<int>();
}

// 2. VERSIÓN COLECTIVA
std::vector<int> blelloch_colectiva(int rank, int num_procesos, std::vector<int> datos)
{
  const int n = static_cast<int>(datos.size());
  const int elementos_locales = n / num_procesos;

  std::vector<int> vector_global = rank == 0 ? std::move(datos) : std::vector<int>(n);
  std::vector<int> vector_local(elementos_locales);

  // Arrays auxiliares para rank 0 (para gestionar los offsets)
  std::vector<int> todas_las_sumas(num_procesos);
  std::vector<int> todos_los_offsets(num_procesos);

  // Repartir datos
  MPI_Scatter(
      rank == 0 ? vector_global.data() : nullptr,
      elementos_locales,
      MPI_INT,
      vector_local.data(),
      elementos_locales,
      MPI_INT,
      0,
      MPI_COMM_WORLD);

  // Calculo de suma local
  int suma_local = 0;
  for (int i = 0; i < elementos_locales; i++)
  {
    suma_local += vector_local[i];
  }

  // Calculo de offsets

  // Recolectar todas las sumas en Rank 0
  MPI_Gather(
      &suma_local, 1, MPI_INT, todas_las_sumas.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Rank 0 calcula los offsets secuencialmente
  if (rank == 0)
  {
    int acumulado = 0;
    for (int i = 0; i < num_procesos; i++)
    {
      todos_los_offsets[i] = acumulado;
      acumulado += todas_las_sumas[i];
    }
  }

  // Repartir los offsets calculados a cada proceso
  int offset_propio = 0;
  MPI_Scatter(
      todos_los_offsets.data(), // array de offsets (solo root)
      1,                        // 1 entero por proceso
      MPI_INT,
      &offset_propio, // donde recibo mi offset
      1,
      MPI_INT,
      0,
      MPI_COMM_WORLD);

  // Calculo final
  blellochSerial(vector_local.data(), elementos_locales);

  for (int i = 0; i < elementos_locales; i++)
  {
    vector_local[i] += offset_propio;
  }

  // Recolectar resultado
  MPI_Gather(
      vector_local.data(),
      elementos_locales,
      MPI_INT,
      rank == 0 ? vector_global.data() : nullptr,
      elementos_locales,
      MPI_INT,
      0,
      MPI_COMM_WORLD);

  return rank == 0 ? std::move(vector_global) : std::vector<int>();
}