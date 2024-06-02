#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

// Function prototypes
unsigned char *julia_rgb(int w, int h, float xl, float xr, float yb, float yt, int start, int end);
int julia_point(int w, int h, float xl, float xr, float yb, float yt, int i, int j);
void tga_write(int w, int h, unsigned char rgb[], char *filename);

int main(int argc, char *argv[]) {
    int rank, size;
    int mult = 12;
    int image_width = 1000 * mult;
    int image_height = 1000 * mult;
    float xl = -1.5, xr = 1.5, yb = -1.5, yt = 1.5;
    int chunk_size = 50;  // Initial chunk size for dynamic task distribution
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("\nJULIA_SET:\n");
        printf("  C version.\n");
        printf("  Plot a version of the Julia set for Z(k+1)=Z(k)^2-0.8+0.156i\n");
    }

    double start_time = MPI_Wtime();

    if (rank == 0) {
        unsigned char *global_rgb = (unsigned char *)malloc(image_width * image_height * 3 * sizeof(unsigned char));
        int current_row = 0;

        // Initially send one task to each worker
        for (int worker_rank = 1; worker_rank < size && current_row < image_height; worker_rank++) {
            int end_row = current_row + chunk_size;
            if (end_row > image_height) {
                end_row = image_height;
            }
            int rows_to_send = end_row - current_row;
            
            // Send task details (start row and number of rows) to worker process
            MPI_Send(&current_row, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);
            MPI_Send(&rows_to_send, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);
            current_row = end_row;
        }

        // Receive results and send new tasks dynamically
        while (current_row < image_height) {
            // Receive results from any worker
            int start_row;
            MPI_Recv(&start_row, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            
            int worker_rank = status.MPI_SOURCE;
            int end_row = start_row + chunk_size;
            if (end_row > image_height) {
                end_row = image_height;
            }
            int rows_received = end_row - start_row;

            MPI_Recv(&global_rgb[start_row * image_width * 3], rows_received * image_width * 3, MPI_UNSIGNED_CHAR, worker_rank, 0, MPI_COMM_WORLD, &status);

            end_row = current_row + chunk_size;
            if (end_row > image_height) {
                end_row = image_height;
            }
            int rows_to_send = end_row - current_row;

            // Send new task to this worker
            MPI_Send(&current_row, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);
            MPI_Send(&rows_to_send, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);
            current_row = end_row;
        }

        // Receive remaining results from workers
        for (int worker_rank = 1; worker_rank < size; worker_rank++) {
            int start_row;
            MPI_Recv(&start_row, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD, &status);
            int end_row = start_row + chunk_size;
            if (end_row > image_height) {
                end_row = image_height;
            }
            int rows_received = end_row - start_row;
            printf("Start row: %d\n", start_row);

            MPI_Recv(&global_rgb[start_row * image_width * 3], rows_received * image_width * 3, MPI_UNSIGNED_CHAR, worker_rank, 0, MPI_COMM_WORLD, &status);
        }

        // Terminate worker processes by sending a special message
        for (int worker_rank = 1; worker_rank < size; worker_rank++) {
            int terminate_signal = -1;
            MPI_Send(&terminate_signal, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);
        }

        // Save the final image
        tga_write(image_width, image_height, global_rgb, "julia_set.tga");
        free(global_rgb);
    } else {
        // Worker processes execute tasks received from the master
        while (1) {
            int start_row, rows_to_compute;
            // Receive task details (start row and number of rows) from master
            MPI_Recv(&start_row, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            if (start_row == -1) {
                break;  // Terminate when no more tasks are sent
            }
            MPI_Recv(&rows_to_compute, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            // Perform Julia set computation for the assigned rows
            unsigned char *local_rgb = julia_rgb(image_width, image_height, xl, xr, yb, yt, start_row, start_row + rows_to_compute);

            // Send computed RGB data back to the master
            MPI_Send(&start_row, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(local_rgb, rows_to_compute * image_width * 3, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
            free(local_rgb);
        }
    }

    double end_time = MPI_Wtime();
    double time_spent = end_time - start_time;

    if (rank == 0) {
        printf("\nJULIA_SET:\n");
        printf("Normal end of execution.\n");
        printf("Execution time %f seconds\n", time_spent);
    }

    MPI_Finalize();
    return 0;
}

// Function to compute RGB values for Julia set
unsigned char *julia_rgb(int w, int h, float xl, float xr, float yb, float yt, int start, int end) {
    unsigned char *rgb = (unsigned char *)malloc((end - start) * w * 3 * sizeof(unsigned char));

    #pragma omp parallel for schedule(static)
    for (int j = start; j < end; j++) {
        for (int i = 0; i < w; i++) {
            int juliaValue = julia_point(w, h, xl, xr, yb, yt, i, j);
            int k = ((j - start) * w + i) * 3;
            rgb[k] = 255 * (1 - juliaValue);
            rgb[k + 1] = 255 * (1 - juliaValue);
            rgb[k + 2] = 255;
        }
    }
    return rgb;
}

// Function to check if a point is in the Julia set
int julia_point(int w, int h, float xl, float xr, float yb, float yt, int i, int j) {
    float ai, ar;
    float ci = 0.156;
    float cr = -0.8;
    int k;
    float t;
    float x, y;

    x = ((float)(w - i - 1) * xl + (float)(i) * xr) / (float)(w - 1);
    y = ((float)(h - j - 1) * yb + (float)(j) * yt) / (float)(h - 1);

    ar = x;
    ai = y;

    for (k = 0; k < 200; k++) {
        t = ar * ar - ai * ai + cr;
        ai = ar * ai + ai * ar + ci;
        ar = t;
        if (1000 < ar * ar + ai * ai) {
            return 0;
        }
    }
    return 1;
}

void tga_write(int w, int h, unsigned char rgb[], char *filename) {
    FILE *file_unit;
    unsigned char header1[12] = { 0,0,2,0,0,0,0,0,0,0,0,0 };
    unsigned char header2[6] = { w%256, w/256, h%256, h/256, 24, 0 };

    // Create the file.
    file_unit = fopen(filename, "wb");

    // Write the headers.
    fwrite(header1, sizeof(unsigned char), 12, file_unit);
    fwrite(header2, sizeof(unsigned char), 6, file_unit);

    // Write the image data.
    fwrite(rgb, sizeof(unsigned char), 3 * w * h, file_unit);

    // Close the file.
    fclose(file_unit);

    printf("\n");
    printf("TGA_WRITE:\n");
    printf("  Graphics data saved as '%s'\n", filename);

    return;
}
