typedef struct{
    int* indices;
    float* points;
    int* neighbor;
    float* k_simplices;
} alpha_complex;

__device__ float calc_sigma(int* indices, float* points)
//circle radius of triangle
{
    float d[3];
    float s = 0;
    for (int i = 0; i<3; i++){
        float p1 = points[indices[i]*2] - points[indices[(i+1)%3]*2];
        float p2 = points[indices[i]*2+1]-points[indices[(i+1)%3]*2+1];
        d[i] = sqrtf(p1*p1+p2*p2);
        s += d[i];}
    s = s/2;
    float area = sqrtf(s*(s-d[0])*(s-d[1])*(s-d[2]));
    float circle_r = d[0]*d[1]*d[2]/(4.0*area);
    return circle_r;
}

__global__ void create_simplices(alpha_complex* complex){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int indices[3];
    int indices2[3];
    float *points = complex->points;
    float *k_sim = complex->k_simplices;

        for (int i = 0; i<3; i++){
            indices[i] = complex->indices[idx*3+i];
        }
        for (int i = 0; i<3; i++){

            k_sim[idx*15 + i*5 + 0] = (float)complex->indices[idx*3+i];
            k_sim[idx*15 + i*5 + 1] = (float)complex->indices[idx*3+(i+1)%3];
            float p1 = points[indices[i]*2] - points[indices[(i+1)%3]*2];
            float p2 = points[indices[i]*2+1] - points[indices[(i+1)%3]*2+1];
            float sigma = sqrtf(p1*p1+p2*p2);
            k_sim[idx*15 + i*5 +2] = sigma;
            if(complex->neighbor[idx*3+(i+2)%3] == -1)
            //only calc one radius if no neighbor
            {
                float dist1 = calc_sigma(indices, points);
                k_sim[idx*15 + i*5 + 3] = fminf(dist1,sigma);
                k_sim[idx*15 + i*5 + 4] = 99999.0;
            }
            else
            //calc radius of nearest neighbor triangles and line distance
            {
                //todo: set neighbor to -1 to avoid double analysis
                for(int j = 0;j<3;j++){
                    indices2[j] = complex->indices[complex->neighbor[idx*3+(i+2)%3]*3+j];
                    //weird indexing from scipy delaunay
                }
                float dist1 = calc_sigma(indices, points);
                float dist2 = calc_sigma(indices2, points);
                if (fminf(dist1, dist2)<1){
                    k_sim[idx*15 +  i*5 + 3] = sigma;
                    }
                else{
                     k_sim[idx*15 +  i*5 + 3] = fminf(dist1, dist2);
                    }
                k_sim[idx*15 +  i*5 + 4] = fmaxf(dist1, dist2);
            }
    }

}