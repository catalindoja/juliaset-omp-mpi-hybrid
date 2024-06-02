#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

int main ( );
unsigned char *julia_rgb ( int w, int h, float xl, float xr, float yb, float yt,int start,int end );
int julia_point ( int w, int h, float xl, float xr, float yb, float yt, int i, 
  int j );
void tga_write ( int w, int h, unsigned char rgb[], char *filename );

/******************************************************************************/



int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int size_factor = 12;
    double start_time = 0;
    double end_time = 0;
    int h = 1000 * size_factor;
    int w = 1000 * size_factor;
    float xl = -1.5;
    float xr = 1.5;
    float yb = -1.5;
    float yt = 1.5;

    if (rank == 0) {
        printf("\n");
        printf("JULIA_SET:\n");
        printf("  C version.\n");
        printf("  Plot a version of the Julia set for Z(k+1)=Z(k)^2-0.8+0.156i\n");
    }

    start_time = MPI_Wtime();

    int chunk_size = h / size;
    int start = rank * chunk_size;
    int end = (rank + 1) * chunk_size;
    if(size<1){
    unsigned char *local_rgb = julia_rgb(w, h, xl, xr, yb, yt, start, end);
    unsigned char *all_rgb = (unsigned char *)malloc(w * h * 3 * sizeof(unsigned char)); // Allocate memory on all processes
    
    MPI_Gather(local_rgb, (end - start) * w * 3, MPI_UNSIGNED_CHAR, all_rgb, (end - start) * w * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        tga_write(w, h, all_rgb, "julia_set.tga");
    }
    free(local_rgb);
    free(all_rgb); // Free memory allocated for all processes
    }else{
      unsigned char *local_rgb = julia_rgb(w, h, xl, xr, yb, yt, start, end);
      tga_write(w, h, local_rgb, "julia_set.tga");
      free(local_rgb);
    }
    end_time = MPI_Wtime();

    // Image writing should be done only on the root process

    

    if (rank == 0) {
        double time_spent = end_time - start_time;
        printf("\n");
        printf("JULIA_SET:\n");
        printf("Normal end of execution.\n");
        printf("Execution time %f seconds\n", time_spent);
    }

    MPI_Finalize();
    return 0;
}
/******************************************************************************/

unsigned char *julia_rgb(int w, int h, float xl, float xr, float yb, float yt, int start, int end)
{
    unsigned char *rgb = (unsigned char *)malloc((end - start) * w * 3 * sizeof(unsigned char));



    #pragma omp parallel for schedule(static)
    for (int j = start; j < end; j++)
    {
        for (int i = 0; i < w; i++)
        {
            int juliaValue = julia_point(w, h, xl, xr, yb, yt, i, j);

            int k = ((j - start) * w + i) * 3;
            rgb[k] = 255 * (1 - juliaValue);
            rgb[k + 1] = 255 * (1 - juliaValue);
            rgb[k + 2] = 255;
        }
    }

    return rgb;
}
/******************************************************************************/

int julia_point ( int w, int h, float xl, float xr, float yb, float yt, int i, int j )

/******************************************************************************/
/*
  Purpose:

    JULIA_POINT returns 1 if a point is in the Julia set.

  Discussion:

    The iteration Z(k+1) = Z(k) + C is used, with C=-0.8+0.156i.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 March 2017

  Parameters:

    Input, int W, H, the width and height of the region in pixels.

    Input, float XL, XR, YB, YT, the left, right, bottom and top limits.

    Input, int I, J, the indices of the point to be checked.

    Ouput, int JULIA, is 1 if the point is in the Julia set.
*/
{
  float ai;
  float ar;
  float ci = 0.156;
  float cr = -0.8;
  int k;
  float t;
  float x;
  float y;
/*
  Convert (I,J) indices to (X,Y) coordinates.
*/
  x = ( ( float ) ( w - i - 1 ) * xl
      + ( float ) (     i     ) * xr ) 
      / ( float ) ( w     - 1 );

  y = ( ( float ) ( h - j - 1 ) * yb
      + ( float ) (     j     ) * yt ) 
      / ( float ) ( h     - 1 );
/*
  Think of (X,Y) as real and imaginary components of
  a complex number A = x + y*i.
*/
  ar = x;
  ai = y;
/*
  A -> A * A + C
*/
  
  for ( k = 0; k < 200; k++ )
  {
    t  = ar * ar - ai * ai + cr;
    ai = ar * ai + ai * ar + ci;
    ar = t;
/*
  if 1000 < ||A||, reject the point.
*/
    if ( 1000 < ar * ar + ai * ai )
    {
      return 0;
    }
  }

  return 1;
}
/******************************************************************************/

void tga_write ( int w, int h, unsigned char rgb[], char *filename )

/******************************************************************************/
/*
  Purpose:

    TGA_WRITE writes a TGA or TARGA graphics file of the data.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    06 March 2017

  Parameters:

    Input, int W, H, the width and height of the image.

    Input, unsigned char RGB[W*H*3], the pixel data.

    Input, char *FILENAME, the name of the file to contain the screenshot.
*/
{
  FILE *file_unit;
  unsigned char header1[12] = { 0,0,2,0,0,0,0,0,0,0,0,0 };
  unsigned char header2[6] = { w%256, w/256, h%256, h/256, 24, 0 };
/* 
  Create the file.
*/
  file_unit = fopen ( filename, "wb" );
/*
  Write the headers.
*/
  fwrite ( header1, sizeof ( unsigned char ), 12, file_unit );
  fwrite ( header2, sizeof ( unsigned char ), 6, file_unit );
/*
  Write the image data.
*/
  fwrite ( rgb, sizeof ( unsigned char ), 3 * w * h, file_unit );
/*
  Close the file.
*/
  fclose ( file_unit );

  printf ( "\n" );
  printf ( "TGA_WRITE:\n" );
  printf ( "  Graphics data saved as '%s'\n", filename );

  return;
}
