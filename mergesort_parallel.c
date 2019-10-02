#define MAXARR 40
#define RANDMAX 40
#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void merge(int a[], int l, int m, int r);
void mergeSort(int a[], int l, int r);
void print_int_array(int int_arr[], int size);

int main(int argc, char *argv[]) {
    int world_rank, world_size, rand_max = RANDMAX, datasize = MAXARR, *data;
    double start,end;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Status status;

    srand(time(NULL));

    start = MPI_Wtime();

    if (world_rank == 0) {
        data = malloc(sizeof(int) * datasize);
        int i;
        for (i = 0; i < datasize; i++) {
            data[i] = rand() % rand_max;
        }
    }
    MPI_Bcast(&datasize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int buffer_size = datasize/world_size;
    int *buffer = malloc(sizeof(int) * buffer_size);
    MPI_Scatter(data, buffer_size, MPI_INT, buffer, buffer_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        free(data);
    }

    mergeSort(buffer, 0, buffer_size-1);

    printf("Rank %d -> ", world_rank);
    print_int_array(buffer, buffer_size);

    int iter_time = (int)log2(world_size);
    int i;
    for (i = 1; i <= iter_time; i++) {
        int leader = world_rank%(int)pow(2,i);
        if (leader != 0) {
            int target = world_rank - (int) pow(2,i-1);
            MPI_Send(buffer, buffer_size, MPI_INT, target, target, MPI_COMM_WORLD);
            free(buffer);
            break;
        }
        else {
            int *recv = malloc(sizeof(int) * buffer_size);
            int previous_size = buffer_size;
            buffer_size = buffer_size*2;
            buffer = realloc(buffer, sizeof(int) * buffer_size);
            int source = world_rank + (int) pow(2, i-1);
            MPI_Recv(recv, previous_size, MPI_INT, source, world_rank, MPI_COMM_WORLD, &status);
            int k, j;
            for (k = previous_size, j = 0; k < buffer_size; k++, j++) {
                buffer[k] = recv[j];
            }
            merge(buffer, 0, previous_size - 1, buffer_size-1);
            printf("Rank %d -> ", world_rank);
            print_int_array(buffer, buffer_size);
        }
    }

    end = MPI_Wtime();

    printf("Process %d Timespent: %.16f\n", world_rank, end-start);
    MPI_Finalize();

    return 0;
}

void merge(int a[], int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    int L[n1], R[n2];

    for (i = 0; i < n1; i++)
        L[i] = a[l + i];
    for (j = 0; j < n2; j++)
        R[j] = a[m + 1+ j];

    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            a[k] = L[i];
            i++;
        }
        else
        {
            a[k] = R[j];
            j++;
        }
        k++;
    }


    while (i < n1)
    {
        a[k] = L[i];
        i++;
        k++;
    }


    while (j < n2)
    {
        a[k] = R[j];
        j++;
        k++;
    }
}


void mergeSort(int a[], int l, int r)
{
    if (l < r)
    {

        int m = l+(r-l)/2;

        mergeSort(a, l, m);
        mergeSort(a, m+1, r);

        merge(a, l, m, r);
    }
}


void print_int_array(int int_arr[], int size) {
    int i;
    printf("[");
    for (i = 0; i < size; i++) {
        printf("%d, ", int_arr[i]);
    }
    printf("]\n");
}