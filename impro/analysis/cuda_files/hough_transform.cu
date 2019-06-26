typedef struct {
    int width;
    int height;
    int stride;
    int* elements;
} matrix;

typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} matrixf;

 __device__ matrix GetSubmatrix(matrix A, int row, int col, int block_size)
{
    matrix Asub;
    Asub.width    = block_size;
    Asub.height   = block_size;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * block_size * row
                                         + block_size * col];
    return Asub;
}

 __device__ matrixf GetSubmatrixf(matrixf A, int row, int col, int block_size)
{
    matrixf Asub;
    Asub.width    = block_size;
    Asub.height   = block_size;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * block_size * row
                                         + block_size * col];
    return Asub;
}


__global__ void create_accum(matrix *accum, matrix *r_table, matrixf *gradient_image)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;;//height of image
  int idy = threadIdx.y + blockDim.y * blockIdx.y;//width of image
  int idz = threadIdx.z + blockDim.z * blockIdx.z;;//width of r_table
  //float phi =0;
  //if(idx<gradient_image->height && idy<gradient_image->width){
  float  phi = gradient_image->elements[idx * gradient_image->width + idy];
  //}
  int slice =0;
  float pi = 3.14159265359;
  if(phi > 0.001||phi< -0.001){
        slice = __float2int_rd(8*(phi+pi)/(2*pi));//rotate here?
        if(r_table->elements[(slice*r_table->width + idz)*2] != 0 && r_table->elements[(slice*r_table->width + idz)*2+1] != 0){

            int ix =  idx+r_table->elements[(slice*r_table->width + idz)*2];
            int iy =  idy+r_table->elements[(slice*r_table->width + idz)*2 + 1];
            if ( ix >= 0 && ix < accum->width && iy >= 0 && iy < accum->height){
               atomicAdd(&accum->elements[(ix*accum->width + iy)],1);
               __syncthreads();
         }
        }
    }
}