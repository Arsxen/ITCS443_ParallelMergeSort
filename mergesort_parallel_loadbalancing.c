#define RANDMAX 100
#define MAXARR 100
#define TOTALBINS 10 //ceil(sqrt(MAXARR)) *must be recalculated when MAXARR is modified!
#define PRTCONSOLE 1 //print final result to terminal or not (0 = false, 1 = true)

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <mpi.h>

//The Struct that use to create the histogram.

typedef struct histogram_bin {
    int lower_bound;
    int upper_bound;
    int frequency;
    int cumulative_freq;
} histogram_bin;

typedef struct histogram {
    int start_bin;
    int end_bin;
    int total_bins;
    histogram_bin bins[TOTALBINS];
} histogram;

typedef struct histogram_options{
    int min;
    int max;
    int width;
} histogram_options;

typedef struct range{
    int min;
    int max;
} range;

void merge(int a[], int l, int m, int r);
void mergeSort(int a[], int l, int r);
void print_int_array(int int_arr[], int size);
void print_histogram_options(histogram_options *ho);
void print_histogram(histogram *hs);
void create_histogram(histogram *dest, const int sorted_arr[], int size,histogram_options *options); //input array must be sorted!
void range_minmax_mergedHistogram(histogram hists[], int size, int *dest_start, int *dest_end);
int *histogram_filtering(int data[], int *ret_size, histogram *hist, range *bin_range);
void destroy_histogram(histogram *hist);
void int_array_copy(int *dest, int *src, int size);

int main(int argc, char *argv[]) {
    int world_rank, world_size;
    int *data = NULL;
    double start,end;
    srand(1234);
    
    //Init MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //Create New Datatype for MPI
    MPI_Datatype dt_bin;
    MPI_Type_contiguous(4, MPI_INT, &dt_bin);
    MPI_Type_commit(&dt_bin);

    MPI_Datatype dt_histogram;
    MPI_Aint displacement[] = {offsetof(histogram, start_bin) ,offsetof(histogram, end_bin),
                               offsetof(histogram, total_bins), offsetof(histogram, bins)};
    int block_length[] = {1, 1, 1, TOTALBINS};
    MPI_Datatype types[] = {MPI_INT, MPI_INT ,MPI_INT, dt_bin};
    MPI_Type_create_struct(4, block_length, displacement, types, &dt_histogram);
    MPI_Type_commit(&dt_histogram);

    MPI_Datatype dt_hoptions;
    MPI_Type_contiguous(3, MPI_INT, &dt_hoptions);
    MPI_Type_commit(&dt_hoptions);

    start = MPI_Wtime();

    histogram_options his_options;
    //Random Number to fill *data
    if (world_rank == 0) {
        data = malloc(sizeof(int) * MAXARR);
        int i, min = RANDMAX + 1, max = -1;
        for (i = 0; i < MAXARR; i++) {
            data[i] = rand() % RANDMAX;
            if (min > data[i]) {
                min = data[i];
            }
            if (max < data[i]) {
                max = data[i];
            }
        }
        int width = ceil((max-min)/(double)TOTALBINS);
        his_options.min = min;
        his_options.max = max;
        his_options.width = width;
    }
    //BroadCast histogram options
    MPI_Bcast(&his_options, 1, dt_hoptions, 0, MPI_COMM_WORLD);

    //Scatter data to all processes
    int buffsize = MAXARR/world_size;
    int *buffer = malloc(sizeof(int) * buffsize);
    MPI_Scatter(data, buffsize, MPI_INT, buffer, buffsize, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank == 0) {
        free(data);
    }

    //Locally Sort
    mergeSort(buffer, 0, buffsize-1);

    int iter_time = (int) log2(world_size);
    int iter_num;
    for (iter_num = 2; iter_num <= iter_time + 1; iter_num++) {

        //Group processes together
        int processes_num = (int) pow(2, iter_num- 1);
        int color = world_rank/processes_num;
        MPI_Comm group_comm;
        MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &group_comm);
        int comm_rank;
        MPI_Comm_rank(group_comm, &comm_rank);

        //Create Histogram
        histogram hist;
        create_histogram(&hist, buffer, buffsize, &his_options);

        //Send Histogram to group
        histogram recv_hist[processes_num];
        MPI_Allgather(&hist, 1, dt_histogram, recv_hist, 1, dt_histogram, group_comm);

        //Merge histogram and get number of bins
        int start_bin;
        int end_bin;
        range_minmax_mergedHistogram(recv_hist, processes_num, &start_bin, &end_bin);
        int bin_num = end_bin - start_bin + 1;
        int bins_width = bin_num/processes_num;

        //Create Range for filtering
        range subrange[processes_num];
        int sr_index;
        int next_start = start_bin;
        int next_end = start_bin + bins_width - 1;
        for (sr_index = 0; sr_index < processes_num; sr_index++) {
            if (sr_index == processes_num - 1) { //last processes in group
                next_end = end_bin;
            }
            subrange[sr_index].min = next_start;
            subrange[sr_index].max = next_end;
            next_start += bins_width;
            next_end = next_start + bins_width - 1;
        }

        //Filtering for sending
        int *send_data[processes_num];
        int send_size[processes_num];
        int f_index;
        for (f_index = 0; f_index < processes_num; f_index++) {
            send_data[f_index] = histogram_filtering(buffer, &send_size[f_index], &hist, &subrange[f_index]);
        }

        //Send Size to all processes
        int recv_size[processes_num];
        MPI_Alltoall(send_size, 1, MPI_INT, recv_size, 1, MPI_INT, group_comm);

        //Free and Allocate new buffer for merging data
        free(buffer);
        int new_size = 0;
        for (f_index = 0; f_index < processes_num; f_index++) {
            new_size += recv_size[f_index];
        }
        buffer = malloc(sizeof(int) * new_size);
        buffsize = new_size;
        int bufflen = send_size[comm_rank];
        int_array_copy(buffer, send_data[comm_rank], bufflen);

        //Prepare buffer
        int *recv_data[processes_num];
        for (f_index = 0; f_index < processes_num; f_index++) {
            recv_data[f_index] = malloc(sizeof(int) * recv_size[f_index]);
        }

        //Async Send Receive
        MPI_Request request[processes_num];
        int t_rank;
        for (t_rank = 0; t_rank < processes_num; t_rank++) {
            if (t_rank != comm_rank) {
                MPI_Isend(send_data[t_rank], send_size[t_rank], MPI_INT, t_rank, comm_rank, group_comm, &request[t_rank]);
            }
        }
        for (t_rank = 0; t_rank < processes_num; t_rank++) {
            if (t_rank != comm_rank) {
                MPI_Irecv(recv_data[t_rank], recv_size[t_rank], MPI_INT, t_rank, t_rank, group_comm, &request[t_rank]);
            }
        }
        for (t_rank = 0; t_rank < processes_num; t_rank++) {
            if (t_rank != comm_rank) {
                MPI_Wait(&request[t_rank], NULL);
                int o;
                int pre_len = bufflen;
                for (o = 0; o < recv_size[t_rank]; o++) {
                    buffer[bufflen] = recv_data[t_rank][o];
                    bufflen++;
                }
                merge(buffer, 0, pre_len-1, bufflen-1);
            }
        }

        //Free Memory from allocation
        int f;
        for (f = 0; f < processes_num; f++) {
            free(send_data[f]);
        }

        destroy_histogram(&hist);

        MPI_Comm_free(&group_comm);
    }

    end = MPI_Wtime();
    printf("World Rank %d Time Spent: %.16f\n", world_rank,end-start);

    if (PRTCONSOLE) {
        printf("[Final] world: %d ->", world_rank);
        print_int_array(buffer, buffsize);
    }

    MPI_Finalize();

    return 0;
}

//Sequential Merge Sort///////////////////
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

/////////////////////////////////////////
void print_int_array(int int_arr[], int size) {
    int i;
    printf("[");
    for (i = 0; i < size; i++) {
        printf("%d, ", int_arr[i]);
    }
    printf("]\n");
}

void print_histogram_options(histogram_options *ho) {
    printf("HOs->Min: %d\n", (*ho).min);
    printf("HOs->MAX: %d\n", (*ho).max);
    printf("HOs->WIDTH: %d\n", (*ho).width);
}

void print_histogram(histogram *hs) {
    int i;
    printf("Total Bins: %d\n", hs->total_bins);
    for (i = hs->start_bin; i <= hs->end_bin; i++) {
        printf("Bin%d min: %d max: %d freq: %d cumu_freq: %d\n", i, hs->bins[i].lower_bound,
               hs->bins[i].upper_bound, hs->bins[i].frequency, hs->bins[i].cumulative_freq);
    }
}

void create_histogram(histogram *dest, const int sorted_arr[], int size, histogram_options *options) {
    //Create bins in histogram with freq = 0
    int i;
    dest->total_bins = TOTALBINS;
    int low = options->min;
    for (i = 0; i < TOTALBINS; i++) {
        dest->bins[i].lower_bound = low;
        int high = low + (options->width -1);
        if (high > options->max) {
            high = options->max;
        }
        dest->bins[i].upper_bound = high;
        dest->bins[i].frequency = 0;
        low += options->width;
    }
    //Count the freq for each bin
    int previous_bin_index = -1;
    int n_bins = 0;
    for (i = 0; i < size; i++) {
        int bin_index = (sorted_arr[i]) / options->width;
        if (previous_bin_index != bin_index)
            n_bins++;
        if (i == 0) {
            dest->start_bin = bin_index;
        }
        else if (i == size - 1) {
            dest->end_bin = bin_index;
        }
        dest->bins[bin_index].frequency++;
        previous_bin_index = bin_index;
    }
    dest->total_bins = n_bins;

    //calculate cumulative frequency
    int previous_cumu_freq = 0;
    for (i = 0; i < TOTALBINS; i++) {
        dest->bins[i].cumulative_freq = previous_cumu_freq + dest->bins[i].frequency;
        previous_cumu_freq = dest->bins[i].cumulative_freq;
    }
}

void range_minmax_mergedHistogram(histogram hists[], int size, int *dest_start, int *dest_end) {
    int start_bin = hists[0].start_bin;
    int end_bin = hists[0].end_bin;
    int i = 0;
    for (i = 0; i < size; i++) {
        if (start_bin > hists[i].start_bin)
            start_bin = hists[i].start_bin;
        if (end_bin < hists[i].end_bin)
            end_bin = hists[i].end_bin;
    }
    *dest_start = start_bin;
    *dest_end = end_bin;
}

int *histogram_filtering(int data[], int *ret_size, histogram *hist, range *bin_range) {
    int totalElem = hist->bins[bin_range->max].cumulative_freq -
                    hist->bins[bin_range->min].cumulative_freq +
                    hist->bins[bin_range->min].frequency;
    int *filter_data = malloc(sizeof(int) * totalElem);
    int i = hist->bins[bin_range->min].cumulative_freq - hist->bins[bin_range->min].frequency;
    int j = 0;
    for (;i < hist->bins[bin_range->max].cumulative_freq; i++,j++) {
        filter_data[j] = data[i];
    }
    *ret_size = totalElem;
    return filter_data;
}

void destroy_histogram(histogram *hist) {
    int i;
    for (i = 0; i < TOTALBINS; i++) {
        hist->bins[i].upper_bound = 0;
        hist->bins[i].lower_bound = 0;
        hist->bins[i].frequency = 0;
    }
    hist->end_bin = 0;
    hist->start_bin = 0;
    hist->total_bins = 0;
}

void int_array_copy(int *dest, int *src, int size) {
    int i;
    for (i = 0; i < size; i++) {
        dest[i] = src[i];
    }
}
