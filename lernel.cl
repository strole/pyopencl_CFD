__kernel void loop1(__global float* psi, int a, int b, int m)
        {
         unsigned int gid = get_global_id(0);
         unsigned int threads = get_global_size(0);
         unsigned int i;
         for(i=gid;i<b-a;i=i+threads){
             psi[(i+a)*(m+2)+0]=(i+a)-(a-1);
             }
         }

__kernel void loop2(__global float* psi, int a, int b, int w)
        {
         unsigned int gid = get_global_id(0);
         unsigned int threads = get_global_size(0);
         unsigned int i;
         for(i=gid;i<b-a;i=i+threads){
             psi[(i+a)*(b+1)+0]=w;
             }
         }

__kernel void loop3(__global float* psi, int h, int m, int w)
        {
         unsigned int gid = get_global_id(0);
         unsigned int threads = get_global_size(0);
                  unsigned int i;
         for(i=gid+1;i<h;i=i+threads){
             psi[(m+1)*(m+2)+i]=w;
             }

         }

__kernel void loop4(__global float* psi, int a, int b, int m, int w)
        {
         unsigned int gid = get_global_id(0);
         unsigned int threads = get_global_size(0);
                  unsigned int i;
         for(i=gid+a;i<b;i=i+threads){
             psi[(m+1)*(m+2)+(i)]=w-i+(b-w);
             }

         }

__kernel void loop5(__global float* psi,int m, int n, __global float* list)
        {
         unsigned int gid = get_global_id(0);
         unsigned int threads = get_global_size(0);
                  unsigned int i,j;
         float bnorm=0;
         for(i=gid; i < m; i+=threads){
            bnorm=0;
            for(j=0; j < n; j++){
            bnorm=bnorm+(psi[i*(m)+j]*psi[i*(m)+j]);
         }
         list[i]=bnorm;
         }
}

__kernel void loop6(__global float* psi,__global float* psitmp, int m, int n)
        {
         unsigned int gid = get_global_id(0);
         unsigned int threads = get_global_size(0);
                  unsigned int i,j;
         for(i=gid+1; i < m; i+=threads){
            for(j=1; j < n; j++){
            psitmp[i*(m+1)+j]=0.25*(psi[(i-1)*(m+1)+j] + psi[(i+1)*(m+1)+j] + psi[i*(m+1)+j-1] + psi[i*(m+1)+j+1]);
         }
         }
}


__kernel void loop7(__global float* psi,__global float* psitmp, int m, int n, __global float* dsq)
        {
         unsigned int gid = get_global_id(0);
         unsigned int threads = get_global_size(0);
         float tmp=0;
         float ddsq=0;
         unsigned int i,j;
         for(i=gid+1; i < m; i+=threads){
         ddsq=0;
            for(j=1; j < n; j++){
            tmp=psitmp[i*(m+1)+j] - psi[i*(m+1)+j];
            ddsq=ddsq+(tmp*tmp);
         }
         dsq[i]=ddsq;
         }
}


__kernel void loop8(__global float* psi,__global float* psitmp, int m, int n)
        {
         unsigned int gid = get_global_id(0);
         unsigned int threads = get_global_size(0);
         unsigned int i,j;
         for(i=gid+1; i < m; i+=threads){
            for(j=1; j < n; j++){
                psi[i*(m+1)+j]=psitmp[i*(m+1)+j];
         }
         }
}

