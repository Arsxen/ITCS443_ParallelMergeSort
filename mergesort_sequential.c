#define RANDMAX 44800
#define MAXARR 200000
#include<stdlib.h>
#include<stdio.h>
#include<mpi.h>

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

int main()
{
//    clock_t begin = clock();
    double start,end;
    start = MPI_Wtime();
	srand(1234);
	int *data = malloc(sizeof(int) * MAXARR);
	int i;
    for (i = 0; i < MAXARR; i++) {
        data[i] = rand() % RANDMAX;
    }
//    clock_t end = clock();
//    printf("Time Spent: %.16f\n", (double)(end - begin)/CLOCKS_PER_SEC);
    end = MPI_Wtime();
    printf("Time Spent: %.16f\n", end-start);

    FILE *fp;
    fp = fopen("Output_Sequential.txt", "w+");
    for (i = 0; i < MAXARR; i++) {
        if (i != MAXARR-1)
            fprintf(fp, "%d,", data[i]);
        else
            fprintf(fp, "%d", data[i]);
    }
    fclose(fp);

	return 0;
}
