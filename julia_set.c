# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <omp.h>

int main ( );
unsigned char *julia_rgb ( int w, int h, float xl, float xr, float yb, float yt );
int julia_point ( int w, int h, float xl, float xr, float yb, float yt, int i, 
  int j );
void tga_write ( int w, int h, unsigned char rgb[], char *filename );

/******************************************************************************/

int main()
{
    int size = 5;
    int h = 1000 * size;
    unsigned char *rgb;
    int w = 1000 * size;
    float xl = -1.5;
    float xr = 1.5;
    float yb = -1.5;
    float yt = 1.5;

    printf("\n");
    printf("JULIA_SET:\n");
    printf("  C version.\n");
    printf("  Plot a version of the Julia set for Z(k+1)=Z(k)^2-0.8+0.156i\n");

    double start_time = omp_get_wtime(); // Start timing using OpenMP timer

    rgb = julia_rgb(w, h, xl, xr, yb, yt);

    double end_time = omp_get_wtime(); // Stop timing using OpenMP timer

    double time_spent = end_time - start_time;

    tga_write(w, h, rgb, "julia_set.tga");

    free(rgb);

    printf("\n");
    printf("JULIA_SET:\n");
    printf("Normal end of execution.\n");
    printf("Execution time %f seconds\n", time_spent);

    return 0;
}
/******************************************************************************/

unsigned char *julia_rgb(int w, int h, float xl, float xr, float yb, float yt)
{
    unsigned char *rgb = (unsigned char *)malloc(w * h * 3 * sizeof(unsigned char));
    
    #pragma omp parallel for schedule(static) shared(rgb,w,h, xl, xr, yb, yt)
    for (int j = 0; j < h; j++)
    {
        for (int i = 0; i < w; i++)
        {
            int juliaValue = julia_point(w, h, xl, xr, yb, yt, i, j);

            int k = (j * w + i) * 3;
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
