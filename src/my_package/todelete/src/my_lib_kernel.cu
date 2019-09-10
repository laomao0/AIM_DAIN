#include <stdbool.h>
#include <stdio.h>

#include "my_lib_kernel.h"

#define min(a,b) ((a<b)?(a):(b))
#define max(a,b) ((a>b)?(a):(b))

#define DEBUG (0)
#ifndef BLOCKDIMX
#define BLOCKDIMX (32)
#endif
#ifndef BLOCKDIMY
#define BLOCKDIMY (16)
#endif


//forward path of our layer
 


//forward path of our layer
__global__ void WeightedFlowProjection_gpu_forward_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel, const float threshhold,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
        const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,
		const int weight_b_stride,const int weight_c_stride, const int weight_h_stride, const int weight_w_stride,

		at::Tensor&  input1,at::Tensor&  input2, at::Tensor&  input3,
		at::Tensor&  count, at::Tensor& weight,
		at::Tensor&  output
		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off = batch_i * input1_b_stride;

	//    __syncthreads();
//	const float fillvalue =0.0f;
//    if (blockIdx.z == 1 && w_i == 32 && h_i == 32){
 //       printf("\nthere is a batch 1\n");
 //   }
	if( withinXbounds && withinYbounds) {
//	    if (blockIdx.z == 1 && w_i == 32 && h_i == 32){
//        printf("\nthere is a batch 1 A\n");
 //   }

        float fx = input1[ off + 0 * input1_c_stride + h_i * input1_h_stride + w_i ];
        float fy = input1[ off + 1 * input1_c_stride + h_i * input1_h_stride + w_i ];

        float x2 = (float) (w_i) + fx;
        float y2 = (float) (h_i) + fy;
        if(x2>=0.0f && y2 >= 0.0f &&x2 <= (float) ( w-1) && y2 <= (float) (h -1 ) ){


			int x3 = (int)(max(min((float) (w_i ) + 2.0f * fx, (float) (w) - 1.0f), 0.0f));//for calculating the brightness constancy between input2 and input3
			int y3 = (int)(max(min((float) (h_i ) + 2.0f * fy, (float) (h) - 1.0f), 0.0f));
			float weight_i = 0.0f;//data1[3],data2[3];
			int channel_i;
			for(channel_i = 0; channel_i < 3; channel_i ++){
				float data1 = input2[batch_i * input2_b_stride + channel_i* input2_c_stride +
											h_i * input2_h_stride + w_i * input2_w_stride];
				float data2 = input3[batch_i * input3_b_stride + channel_i *input3_c_stride +
										y3 * input3_h_stride + x3 * input3_w_stride];
				weight_i += fabs(data1 - data2)/3.0f;
	//			    if (blockIdx.z == 1 && w_i == 32 && h_i == 32){
///				    printf("\n%d,%d, %f,%f,%f\n" , x3,y3, data1,data2,weight_i);
	//			    }
			}
			weight_i += 1e-8f; //add a small constant for better verification
    //if (blockIdx.z == 1 && w_i == 32 && h_i == 32){
     //   printf("\nthere is a batch 1 B, weight i is %f, threshold is %f\n", weight_i, threshhold);
   // }
			if(weight_i <= threshhold){
 //   if (blockIdx.z == 1 && w_i == 32 && h_i == 32){
  //      printf("\nbatch 1 is processed\n");
   // }

            int ix2_L = (int) (x2);
            int iy2_T = (int) (y2);
            int ix2_R = min(ix2_L + 1, w - 1);
            int iy2_B = min(iy2_T + 1, h - 1);

            atomicAdd(&output[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L ],-fx);
            atomicAdd(&output[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ],-fx);
            atomicAdd(&output[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L ],-fx);
            atomicAdd(&output[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R ],-fx);

            atomicAdd(&output[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L]  , -fy);
            atomicAdd(&output[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R]  , -fy);
            atomicAdd(&output[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L]  , -fy);
            atomicAdd(&output[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R]  , -fy);

            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L], 1);
            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R] , 1);
            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L] , 1);
            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R] , 1);

			atomicAdd(& weight[batch_i * weight_b_stride + 0 + iy2_T * weight_h_stride + ix2_L] , weight_i);
            atomicAdd(& weight[batch_i * weight_b_stride + 0 + iy2_T * weight_h_stride + ix2_R] , weight_i);
            atomicAdd(& weight[batch_i * weight_b_stride + 0 + iy2_B * weight_h_stride + ix2_L] , weight_i);
            atomicAdd(& weight[batch_i * weight_b_stride + 0 + iy2_B * weight_h_stride + ix2_R] , weight_i);

			}
        }
	}
	return ;

}

__global__ void WeightedFlowProjectionAveraging_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
        const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,
		const int weight_b_stride,const int weight_c_stride, const int weight_h_stride, const int weight_w_stride,

		at::Tensor&  input1,
		at::Tensor&  count,at::Tensor& weight,
		at::Tensor&  output
		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off = batch_i * input1_b_stride;

	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {
	    float temp =count[batch_i * count_b_stride + 0 + h_i * count_h_stride + w_i] ;
        if(temp > 0.0f){
            output[off + 0 * input1_c_stride + h_i * input1_h_stride + w_i ] /= temp;
            output[off + 1 * input1_c_stride + h_i * input1_h_stride + w_i ] /= temp;

			weight[batch_i * weight_b_stride + 0 + h_i * weight_h_stride + w_i ] /= temp;
        }
	}
	return ;

}


__global__ void WeightedFlowFillhole_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,
		const int weight_b_stride,const int weight_c_stride, const int weight_h_stride, const int weight_w_stride,

		at::Tensor&  input1,
		at::Tensor&  count,at::Tensor& weight,
		at::Tensor&  output
		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off = batch_i * input1_b_stride;

	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {
	    float temp = count[batch_i * count_b_stride + 0 + h_i * count_h_stride + w_i] ;
        if(temp <= 0.0f){
            //search along the four directions,0/90/180/270, until finding at least one
            int left_offset = w_i;            float left_temp = 0.0f;
            while(left_temp == 0.0f && left_offset - 1 >= 0){
                left_offset = left_offset - 1;
                left_temp = count[batch_i * count_b_stride + 0 + h_i * count_h_stride + left_offset] ;
            }

            int right_offset = w_i ;            float right_temp = 0.0f;
            while(right_temp ==0.0f && right_offset + 1 <= w - 1 ){
                right_offset  = right_offset + 1 ;
                right_temp =  count[batch_i * count_b_stride + 0 + h_i * count_h_stride + right_offset] ;
            }

            int up_offset = h_i ;            float up_temp = 0.0f;
            while(up_temp == 0.0f && up_offset - 1 >=0){
                up_offset = up_offset - 1;
                up_temp =  count[batch_i * count_b_stride + 0 + up_offset * count_h_stride + w_i ] ;
            }

            int down_offset = h_i;            float down_temp = 0.0f;
            while(down_temp = 0.0f && down_offset + 1 <= h - 1 ){
                down_offset = down_offset + 1;
                down_temp =  count[batch_i * count_b_stride + 0 + down_offset * count_h_stride + w_i] ;
            }

            if(left_temp + right_temp + up_temp + down_temp <=0.0f){
                //printf("Can't fill hole, find no neighbor vectors availabel\n");
                return;
            }

            left_temp = (left_temp > 0.0f)?1:0;
            right_temp = (right_temp > 0.0f)?1:0;
            up_temp = (up_temp > 0.0f)?1:0;
            down_temp = (down_temp > 0.0f)?1:0;

            output[off + 0 * input1_c_stride + h_i * input1_h_stride + w_i ] = (
                left_temp *  output[off + 0 * input1_c_stride + h_i * input1_h_stride + left_offset] +
                right_temp *  output[off + 0 * input1_c_stride + h_i * input1_h_stride + right_offset]+
                up_temp *  output[off + 0 * input1_c_stride + up_offset * input1_h_stride + w_i] +
                down_temp *  output[off + 0 * input1_c_stride + down_offset * input1_h_stride + w_i]
            )/(
                left_temp + right_temp + up_temp + down_temp
            ) ;


            output[off + 1 * input1_c_stride + h_i * input1_h_stride + w_i ] =(
                left_temp *  output[off + 1 * input1_c_stride + h_i * input1_h_stride + left_offset] +
                right_temp *  output[off + 1 * input1_c_stride + h_i * input1_h_stride + right_offset]+
                up_temp *  output[off + 1 * input1_c_stride + up_offset * input1_h_stride + w_i] +
                down_temp *  output[off + 1 * input1_c_stride + down_offset * input1_h_stride + w_i]
            )/(
                left_temp + right_temp + up_temp + down_temp
            ) ;
        }
	}
	return ;

}
__global__ void WeightedFlowProjection_gpu_backward_kernelfunc(
		const int nElement,  	const int w, 	const int h, const int channel, const float threshhold,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
        const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,
		const int weight_b_stride,const int weight_c_stride, const int weight_h_stride, const int weight_w_stride,

		at::Tensor&  input1,at::Tensor&  input2, at::Tensor&  input3,
		at::Tensor&  count,const at::Tensor& weight,
		at::Tensor&  gradoutput,
		at::Tensor&  gradinput1
		)
{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off  = batch_i * input1_b_stride;

	//    __syncthreads();

	if(withinXbounds && withinYbounds){
        float fx = input1[off + 0 * input1_c_stride + h_i * input1_h_stride + w_i] ;
        float fy = input1[off + 1 * input1_c_stride + h_i * input1_h_stride + w_i] ;

        float x2 = (float) ( w_i ) + fx;
        float y2 = (float) ( h_i ) + fy;
        if( x2 >=0.0f && y2 >= 0.0f && x2 <= (float) (w -1) && y2 <= (float) (h-1)){

			int x3 = (int)(max(min((float) (w_i ) + 2.0f * fx, (float) (w) - 1.0f), 0.0f));//for calculating the brightness constancy between input2 and input3
			int y3 = (int)(max(min((float) (h_i ) + 2.0f * fy, (float) (h) - 1.0f), 0.0f));
			float weight_i = 0.0f;//data1[3],data2[3];
			int channel_i;
			for(channel_i = 0; channel_i < 3; channel_i ++){
				float data1 = input2[batch_i * input2_b_stride + channel_i* input2_c_stride +
											h_i * input2_h_stride + w_i * input2_w_stride];
				float data2 = input3[batch_i * input3_b_stride + channel_i *input3_c_stride +
									y3 * input3_h_stride + x3 * input3_w_stride];
				weight_i += fabs(data1 - data2)/3.0f;
			}
			weight_i += 1e-8f; //add a small constant for better verification

			if(weight_i <= threshhold){


            int ix2_L = (int)(x2);
            int iy2_T = (int)(y2);
            int ix2_R  = min(ix2_L + 1, w-1);
            int iy2_B  = min(iy2_T + 1, h-1);

            int iu_offset = off + 0 * input1_c_stride + h_i * input1_h_stride + w_i;
            gradinput1[iu_offset] += -  gradoutput[off +  0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L]/
                                        count[batch_i * count_b_stride + 0+ iy2_T * count_h_stride + ix2_L]  ;
            gradinput1[iu_offset] += -    gradoutput[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ]/
                                         count[batch_i * count_b_stride +0 + iy2_T * count_h_stride  + ix2_R]          ;
            gradinput1[iu_offset ] += -  gradoutput[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L]/
                                         count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L]  ;
            gradinput1[iu_offset ]  += -  gradoutput[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R]/
                                         count[batch_i * count_b_stride + 0+ iy2_B * count_h_stride + ix2_R]   ;

            int iv_offset = off + 1 * input1_c_stride + h_i * input1_h_stride + w_i;
            gradinput1[iv_offset] += -  gradoutput[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L]/
                                         count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L]  ;
            gradinput1[iv_offset] += - gradoutput[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R]/
                                         count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R]  ;
            gradinput1[iv_offset] += -  gradoutput[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L]/
                                    count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L]     ;
            gradinput1[iv_offset] += -  gradoutput[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R]/
                                    count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R]   ;
			}
        }
	}
	return ;

}



int WeightedFlowProjection_gpu_forward_kernel(
		cudaStream_t stream, 		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const int fillhole, const float threshold,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
        const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,
		const int weight_b_stride,const int weight_c_stride, const int weight_h_stride, const int weight_w_stride,

		at::Tensor&  input1, at::Tensor&  input2, at::Tensor&  input3,
		at::Tensor&  count,  at::Tensor& weight,
		at::Tensor&  output
		)
{
    int error = 1 ;


	dim3 grid;
	dim3 block;


	//		blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
  //  printf("I am here, grid size %d, %d, %d\n", grid.x, grid.y, grid.z);

    //printf("\ninput2 stride %d,%d,%d,%d", 						input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride);
    //printf("\ninput3 stride %d,%d,%d,%d", 									input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);
//    printf("\ncount stride %d,%d,%d,%d", 			count_b_stride,count_c_stride,count_h_stride,count_w_stride);
  //  printf("\nweight stride %d,%d,%d,%d", 						weight_b_stride,weight_c_stride,weight_h_stride,weight_w_stride);
	//extract the data of CudaTensor and use kernel to calculate.
	WeightedFlowProjection_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,channel, threshold,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,
			weight_b_stride,weight_c_stride,weight_h_stride,weight_w_stride,

			input1, input2, input3, count, weight, output
			);
    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}
//    printf("I am there\n");

    WeightedFlowProjectionAveraging_kernelfunc<<<grid,block,0,stream>>>(
    		nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,
			weight_b_stride,weight_c_stride,weight_h_stride,weight_w_stride,

			input1,count, weight, output
    );
//    printf("I am kao\n");

	//			THCudaCheck(cudaGetLastError());
    err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}
//    printf("I am dd\n");

    if(fillhole){

//        printf("use flow fill hole\n");
        WeightedFlowFillhole_kernelfunc<<<grid,block,0,stream>>>(
    		nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,
			weight_b_stride,weight_c_stride,weight_h_stride,weight_w_stride,

			input1,count,weight, output
        );

    err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		return error;
	}

    }

	error = 0;
	return error;

}

int WeightedFlowProjection_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int batch, const float threshhold,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
        const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,
		const int	weight_b_stride,const int weight_c_stride, const int weight_h_stride, const int weight_w_stride,

		at::Tensor&  input1,at::Tensor&  input2, at::Tensor&  input3,
		at::Tensor&  count, const at::Tensor&  weight,
		at::Tensor&  gradoutput,
		at::Tensor&  gradinput1
		)
{

	int error = 1 ;

	dim3 grid;
	dim3 block;

	//blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	WeightedFlowProjection_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,channel, threshhold,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,
			weight_b_stride,weight_c_stride,weight_h_stride,weight_w_stride,

			input1, input2, input3,
			count, weight,
			gradoutput,
			gradinput1
			);
//    printf("gpu I am there\n");

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpu error in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}
//    printf("gpu I am here\n");

	error = 0;
	return error;


}


//forward path of our layer
__global__ void WeightLayer_gpu_forward_kernelfunc(
		const int nElement,
		const int w, 		const int h, 		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
	//	const int flow1_grad_b_stride,const int flow1_grad_c_stride,const int flow1_grad_h_stride,const int flow1_grad_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor&  input1, at::Tensor&  input2, at::Tensor&  input3,
		 //const at::Tensor&  flow1_grad,
		 at::Tensor&  output,
		float  lambda_e, float	lambda_v, float Nw

		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int offset = batch_i * input1_b_stride;

	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {
	//read the opticalflow
	float fx = input3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
	float fy = input3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];

	//get the destination position
	float x2 = (float)(w_i) + fx;
	float y2 = (float)(h_i) + fy;

	//Guarrantee that the center position is in-border.
	if(x2 >= 0.0f && y2 >=0.0f && x2 <=  (float) (w -1) && y2  <= (float) (h-1)){
		int ix2_L = (int)(x2);
		int iy2_T = (int)(y2);
		int ix2_R = min(ix2_L+1, w - 1);
		int iy2_B = min(iy2_T+1, h - 1);

		
		float alpha = x2 - (int)(x2);
		float beta =  y2 - (int)(y2);
		
		int m;
		int n;
		
		float err_sum = 0.0f;
	//	float sv_sum = 0.0f ;
		
		// Nw must be 3, so that -1,0,1 is the range
		for(m = -1; m <= 1; m ++){
			int patch1_m = min(max(0, m + h_i), h-1);
			for(n = -1; n <= 1; n ++){
				int patch1_n = min(max(0, n + w_i), w-1);
				
				int patch2_mT = min(max(0, m + iy2_T), h-1);
				int patch2_nL = min(max(0, n + ix2_L), w-1);
				int patch2_mB = min(max(0, m + iy2_B), h-1);
				int patch2_nR = min(max(0, n + ix2_R), w-1);
				for ( int c_i = 0; c_i < channel; c_i ++){											
					float taget_data = 
					(1-alpha)*(1-beta)*input2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nL] +
						alpha*(1-beta)*input2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nR] +
						(1-alpha)*beta*input2[offset + c_i * input2_c_stride + patch2_mB * input2_h_stride + patch2_nL] +
							alpha*beta*input2[offset + c_i * input2_c_stride + patch2_mB * input2_h_stride + patch2_nR];
					
					err_sum += fabsf(input1[offset + c_i * input1_c_stride + patch1_m * input1_h_stride + patch1_n]
								   - taget_data);																
				}
				//sv_sum += flow1_grad[batch_i * flow1_grad_b_stride + 0 + patch1_m * flow1_grad_h_stride + patch1_n];
			}
		}
		err_sum /= (channel * Nw * Nw);
		//sv_sum /= (Nw * Nw);
		output[batch_i * output_b_stride + 0 + h_i * output_h_stride + w_i * output_w_stride]
		    					    = (1-err_sum/lambda_e)*(1-err_sum/lambda_e);
			//= expf( - err_sum/lambda_e - sv_sum/lambda_v);
		
	}
	else {
		output[batch_i * output_b_stride + 0 + h_i * output_h_stride + w_i * output_w_stride]
			= 1e-4f; //this dosen't mean that
	}
	}
	return ;

}
int WeightLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int flow1_grad_b_stride,const int flow1_grad_c_stride,const int flow1_grad_h_stride,const int flow1_grad_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor&  input1,    		at::Tensor&  input2,    	at::Tensor&  input3,
		//const at::Tensor&  flow1_grad,
			at::Tensor&  output,
		float  lambda_e, float	lambda_v, float Nw
		)
{
		int error = 1 ;

	dim3 grid;
	dim3 block;


	//		blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	//extract the data of CudaTensor and use kernel to calculate.
	WeightLayer_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			//flow1_grad_b_stride,flow1_grad_c_stride,flow1_grad_h_stride,flow1_grad_w_stride,
			output_b_stride,  output_c_stride,   output_h_stride,   output_w_stride,

			input1,input2,input3,
			 //flow1_grad,
			 output,
			lambda_e,  lambda_v,   Nw
			);

	//			THCudaCheck(cudaGetLastError());
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;

}



__global__ void WeightLayer_gpu_backward_kernelfunc(
		const int nElement, 	   const int w, 		const int h, 		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int flow1_grad_b_stride,const int flow1_grad_c_stride,const int flow1_grad_h_stride,const int flow1_grad_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor&  input1,     	at::Tensor&  input2, at::Tensor&  input3,
		//const at::Tensor&  flow1_grad,
		at::Tensor&  output,
		at::Tensor&  gradoutput, at::Tensor&  gradinput1, at::Tensor&  gradinput2, at::Tensor&  gradinput3,
		 //at::Tensor&  gradflow1_grad,
		
		float  lambda_e, float	lambda_v, float Nw

		)
		{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int offset  = batch_i * input1_b_stride;

	//    __syncthreads();

	if(withinXbounds && withinYbounds) {
	//read the opticalflow
	float fx = input3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
	float fy = input3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];

	//get the destination position
	float x2 = (float)(w_i) + fx;
	float y2 = (float)(h_i) + fy;

	//Guarrantee that the center position is in-border.
	if(x2 >= 0.0f && y2 >=0.0f && x2 <=  (float) (w -1) && y2  <= (float) (h-1)){
		int ix2_L = (int)(x2);
		int iy2_T = (int)(y2);
		int ix2_R = min(ix2_L+1, w - 1);
		int iy2_B = min(iy2_T+1, h - 1);
		float alpha = x2 - (int)(x2);
		float beta =  y2 - (int)(y2);
	 
		float gradoutput_data_value = gradoutput[batch_i * output_b_stride + 0 + h_i * output_h_stride + w_i * output_w_stride];

		float grad_err_sum = -  gradoutput_data_value / (lambda_e * channel * Nw *Nw)
									* 2 *sqrtf(output [batch_i * output_b_stride + 0 + h_i * output_h_stride + w_i * output_w_stride]) ;
//
		//float grad_err_sum = -  gradoutput_data_value *
		    //                output [batch_i * output_b_stride + 0 + h_i * output_h_stride + w_i * output_w_stride] /
		     //              (lambda_e * channel * Nw *Nw) ;
		//float grad_sv_sum =  - gradoutput_data_value *
		//                    output[batch_i * output_b_stride + 0 + h_i * output_h_stride + w_i * output_w_stride]  /
		//                    (lambda_v * Nw * Nw) ;
		
		int m;
		int n;
		// Nw must be 3, so that -1,0,1 is the range
		for(m = -1; m <= 1; m ++){
			int patch1_m = min(max(0, m + h_i), h-1);
			for(n = -1; n <= 1; n ++){
				int patch1_n = min(max(0, n + w_i), w-1);
											
				int patch2_mT = min(max(0, m + iy2_T), h-1);
				int patch2_nL = min(max(0, n + ix2_L), w-1);
				int patch2_mB = min(max(0, m + iy2_B), h-1);
				int patch2_nR = min(max(0, n + ix2_R), w-1);
				
				for (int c_i = 0; c_i < channel; c_i ++){		 	
					float taget_data =
					(1-alpha)*(1-beta)*input2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nL] +
						alpha*(1-beta)*input2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nR] +
						(1-alpha)*beta*input2[offset + c_i * input2_c_stride + patch2_mB * input2_h_stride + patch2_nL] +
							alpha*beta*input2[offset + c_i * input2_c_stride + patch2_mB * input2_h_stride + patch2_nR] ;
					
					float i_data = input1[offset + c_i * input1_c_stride + patch1_m * input1_h_stride + patch1_n] ;

					//input1 gradients
					atomicAdd(& gradinput1[offset + c_i * input1_c_stride + patch1_m * input1_h_stride + patch1_n] ,
										( i_data > taget_data) ?  grad_err_sum : - grad_err_sum);
					 
					 //input2 gradients
					atomicAdd(& gradinput2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nL] ,
					(1-alpha)*(1-beta)*(( i_data> taget_data) ? - grad_err_sum :  grad_err_sum) );	
					 
					atomicAdd(& gradinput2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nR] ,
					alpha*(1-beta)*( ( i_data> taget_data) ?  - grad_err_sum :  grad_err_sum));	
					 
					atomicAdd(& gradinput2[offset + c_i * input2_c_stride + patch2_mB  * input2_h_stride + patch2_nL] ,
					(1-alpha)*beta*( ( i_data > taget_data) ?   - grad_err_sum :  grad_err_sum));	
					
					atomicAdd(& gradinput2[offset + c_i * input2_c_stride + patch2_mB * input2_h_stride + patch2_nR]  ,
					alpha*beta*( ( i_data > taget_data) ?   - grad_err_sum :  grad_err_sum));									
					 
					//input3 gradients
					float gamma  = 1.0f - beta; //iy2_B - y2;
					float temp = 0.0f;
					temp += gamma * (input2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nR]-
							input2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nL]);
					temp += (1-gamma) *( input2[offset + c_i * input2_c_stride + patch2_mB * input2_h_stride + patch2_nR]  -
										input2[offset + c_i * input2_c_stride + patch2_mB  * input2_h_stride + patch2_nL]);
					
					temp = temp * ( ( i_data > taget_data) ? - grad_err_sum : grad_err_sum);
					atomicAdd(& gradinput3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i] ,
						temp);
						
					gamma = 1.0f - alpha; //ix2_R -x2;
					temp = 0.0f;
					temp += gamma * ( input2[offset + c_i * input2_c_stride + patch2_mB  * input2_h_stride + patch2_nL] - 
										input2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nL]);
					temp += gamma *(input2[offset + c_i * input2_c_stride + patch2_mB * input2_h_stride + patch2_nR]  -
									input2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nR] );
					temp = temp * ( ( i_data > taget_data) ? - grad_err_sum : grad_err_sum);
					atomicAdd(& gradinput3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i] ,
						temp);														
				}
				//flow1_grad's gradients
				//sv_sum += flow1_grad[batch_i * flow1_grad_b_stride + 0 + patch1_m * flow1_grad_h_stride + patch1_n];
				//atomicAdd(& gradflow1_grad[ batch_i * flow1_grad_b_stride + 0 + patch1_m * flow1_grad_h_stride + patch1_n]  ,
				//	grad_sv_sum);
			}
		}		
	}	
	}	
	return ;

}

int WeightLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
//		const int flow1_grad_b_stride,const int flow1_grad_c_stride,const int flow1_grad_h_stride,const int flow1_grad_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor&  input1, at::Tensor&  input2, at::Tensor&  input3,
		 //const at::Tensor&  flow1_grad,
		 at::Tensor&  output,

		at::Tensor&  gradoutput, at::Tensor&  gradinput1, at::Tensor&  gradinput2, at::Tensor&  gradinput3,
		//at::Tensor&  gradflow1_grad,
		float  lambda_e, float	lambda_v, float Nw

		)
{
	int error = 1 ;

	dim3 grid;
	dim3 block;


	//blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);

//    cudaMemset((void*)gradinput1, 0, input1_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput2, 0, input2_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput3, 0, input3_b_stride * batch * sizeof(float));

	WeightLayer_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
//			flow1_grad_b_stride,flow1_grad_c_stride,flow1_grad_h_stride,flow1_grad_w_stride,
			output_b_stride,  output_c_stride,   output_h_stride,   output_w_stride,

			input1, 			input2,         input3,
			//flow1_grad,
			 output,
			gradoutput,
			gradinput1, 			gradinput2,     gradinput3,
			//gradflow1_grad,
			lambda_e,  lambda_v,   Nw
			);

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;	
}

//forward path of our layer
__global__ void PixelValueLayer_gpu_forward_kernelfunc(
		const int nElement,
		const int w, 		const int h, 		const int channel,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor&  input1, at::Tensor&  input3, const at::Tensor&  flow_weights,	at::Tensor&  output,
		float	sigma_d,     float tao_r , float  Prowindow

		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int offset = batch_i * input1_b_stride;

 

	if( withinXbounds && withinYbounds) {
		//read the opticalflow
		float fx = input3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
		float fy = input3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];
		
		//get the destination position
		float x2 = (float)(w_i) + fx/2.0f; //the intermediate position
		float y2 = (float)(h_i) + fy/2.0f;

		//Guarrantee that the center position is in-border.
		if(x2 >= 0.0f && y2 >=0.0f && x2 <=  (float) (w -1) && y2  <= (float) (h-1)){
			int ix2_L = (int)(x2);
			int iy2_T = (int)(y2); 
			
			float alpha = x2 - (int)(x2);
			float beta =  y2  - (int)(y2);
			
			int m;
			int n;
			// we interpolate 4 pixels, should we change the sigma_d ?
			for( m = -1; m <= 2; m ++){
 				for(n = -1; n <= 2; n ++){
 					int patch2_m = min(max(0, m + iy2_T), h-1);
					int patch2_n = min(max(0, n + ix2_L), w-1);
					
//					float g_d =  expf( - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2 * sigma_d * sigma_d));
                      float g_d = (1.0f - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                            g_d = g_d * g_d;
					float f_w = flow_weights[batch_i * flow_weights_b_stride + 0 + h_i * flow_weights_h_stride + w_i]  ;
					for( int c_i = 0 ; c_i < channel; c_i ++){								
						atomicAdd(& output[offset + c_i * output_c_stride +  patch2_m * output_h_stride + patch2_n] ,
							f_w * g_d * input1[offset + c_i * input1_c_stride + h_i * input1_h_stride + w_i]);
					}
				}												
			}											
		}
	
	}
	return ;

}
int PixelValueLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor&  input1, at::Tensor&  input3, const at::Tensor&  flow_weights,	at::Tensor&  output,
		float	sigma_d,     float tao_r , float  Prowindow
		)
{
	int error = 1 ;

	dim3 grid;
	dim3 block;


	// blockthread = 128;
	// the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	// the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	//extract the data of CudaTensor and use kernel to calculate.
	PixelValueLayer_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,

			input1, input3, flow_weights, output,
			sigma_d,      tao_r ,   Prowindow
			);

	//			THCudaCheck(cudaGetLastError());
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;

}



__global__ void PixelValueLayer_gpu_backward_kernelfunc(
		const int nElement, 	   const int w, 		const int h, 		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor&  input1,  at::Tensor&  input3, const at::Tensor&  flow_weights, 
		at::Tensor&  gradoutput, at::Tensor&  gradinput1,   at::Tensor&  gradinput3, at::Tensor&  gradflow_weights,
		
		float	sigma_d,     float tao_r , float  Prowindow

		)
		{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int offset  = batch_i * input1_b_stride;
 
	//    __syncthreads();

	if(withinXbounds && withinYbounds) {

		//read the opticalflow
		float fx = input3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
		float fy = input3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];
		
		//get the destination position
		float x2 = (float)(w_i) + fx/2.0f; //the intermediate position
		float y2 = (float)(h_i) + fy/2.0f;

		//Guarrantee that the center position is in-border.
		if(x2 >= 0.0f && y2 >=0.0f && x2 <=  (float) (w -1) && y2  <= (float) (h-1)){
			int ix2_L = (int)(x2);
			int iy2_T = (int)(y2);
   
 			float alpha = x2 - (int)(x2);
			float beta =  y2  - (int)(y2);
			
			int m;
			int n;
			// we interpolate 4 pixels, should we change the sigma_d ?
			for( m = -1; m <= 2; m ++){
 				for(n = -1; n <= 2; n ++){
 					
					int patch2_m = min(max(0, m + iy2_T), h-1);
					int patch2_n = min(max(0, n + ix2_L), w-1);
					
//					float g_d =  expf( - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                      float g_d = (1.0f - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                      //      g_d = g_d * g_d;
					float f_w = flow_weights[batch_i * flow_weights_b_stride + 0 + h_i * flow_weights_h_stride + w_i]  ;
					
					for(int c_i = 0 ; c_i < channel; c_i ++){	
						float gradoutput_data_value = gradoutput[offset + c_i * input1_c_stride +  patch2_m * input1_h_stride + patch2_n];
                            //input1 gradients
						atomicAdd(& gradinput1[offset + c_i * input1_c_stride + h_i * input1_h_stride + w_i],
							gradoutput_data_value * f_w * g_d*g_d);
						
						// flow_weights_data gradients
						atomicAdd(& gradflow_weights[batch_i * flow_weights_b_stride + 0+ h_i * flow_weights_h_stride + w_i],
							gradoutput_data_value * g_d * g_d * input1[offset + c_i * input1_c_stride + h_i * input1_h_stride + w_i]);
							
						//flow gradients
						atomicAdd(& gradinput3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i] ,
								- gradoutput_data_value * f_w * input1[offset + c_i * input1_c_stride + h_i * input1_h_stride + w_i] *
								 g_d * (n - alpha) / (  sigma_d * sigma_d) * 2.0f);
						atomicAdd(& gradinput3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i] ,
                                - gradoutput_data_value * f_w * input1[offset + c_i * input1_c_stride + h_i * input1_h_stride + w_i] *
								 g_d * (m - beta) / (  sigma_d * sigma_d) * 2.0f);
					}
				}												
			}											
		}
			
	}	
	return ;

}
int PixelValueLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor&  input1,  at::Tensor&  input3, const at::Tensor&  flow_weights, 

		at::Tensor&  gradoutput, at::Tensor&  gradinput1,   at::Tensor&  gradinput3, at::Tensor&  gradflow_weights,
		float	sigma_d,     float tao_r , float  Prowindow

		)
{ 
	int error = 1 ;

	dim3 grid;
	dim3 block;


	//blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);

//    cudaMemset((void*)gradinput1, 0, input1_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput2, 0, input2_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput3, 0, input3_b_stride * batch * sizeof(float));

	PixelValueLayer_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,


			input1, 			      input3,  		flow_weights, 
			gradoutput,
			gradinput1, 			     gradinput3,	gradflow_weights,
			sigma_d,      tao_r ,   Prowindow

			);

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;	
}	


//forward path of our layer
__global__ void PixelWeightLayer_gpu_forward_kernelfunc(
		const int nElement,
		const int w, 		const int h,
		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor&  input3, const at::Tensor&  flow_weights,	at::Tensor&  output,
		float	sigma_d,     float tao_r , float  Prowindow
		)
{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
 
	if( withinXbounds && withinYbounds) {

		//read the opticalflow
		float fx = input3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
		float fy = input3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];
		
		//get the destination position
		float x2 = (float)(w_i) + fx/2.0f; //the intermediate position
		float y2 = (float)(h_i) + fy/2.0f;

		//Guarrantee that the center position is in-border.
		if(x2 >= 0.0f && y2 >=0.0f && x2 <=  (float) (w -1) && y2  <= (float) (h-1)){
			int ix2_L = (int)(x2);
			int iy2_T = (int)(y2);
			
			float alpha = x2 - (int)(x2);
			float beta =  y2  - (int)(y2);
			
			int m;
			int n;
			// we interpolate 4 pixels, should we change the sigma_d ?
			for( m = -1; m <= 2; m ++){ 									
				for(n = -1; n <= 2; n ++){ 					
					int patch2_m = min(max(0, m + iy2_T), h-1);
					int patch2_n = min(max(0, n + ix2_L), w-1);
					
//					float g_d = expf( - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                      float g_d = (1.0f - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                            g_d = g_d * g_d;
					float f_w = flow_weights[batch_i * flow_weights_b_stride + 0 + h_i * flow_weights_h_stride + w_i]  ;
 					atomicAdd(& output[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n] ,  f_w * g_d);
 				}												
			}											
		}	
	}
	return ;

}
int PixelWeightLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 			const int batch,

		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor&  input3, const at::Tensor&  flow_weights,	at::Tensor&  output,
		float	sigma_d,     float tao_r , float  Prowindow
		)
{
		int error = 1 ;

	dim3 grid;
	dim3 block;


	// blockthread = 128;
	// the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	// the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	//extract the data of CudaTensor and use kernel to calculate.
	PixelWeightLayer_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,
			//input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,

			input3, flow_weights, output,
			sigma_d,  tao_r,   Prowindow
			);

	//			THCudaCheck(cudaGetLastError());
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;
}



__global__ void PixelWeightLayer_gpu_backward_kernelfunc(
		const int nElement, 	   const int w, 		const int h,
		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor&  input3, const at::Tensor&  flow_weights,  at::Tensor&  output,
		at::Tensor&  gradoutput, at::Tensor&  gradinput3, at::Tensor&  gradflow_weights,
        float threshhold,
		float	sigma_d,     float tao_r , float  Prowindow
		)
{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
 
 
	if(withinXbounds && withinYbounds) {
		//read the opticalflow
		float fx = input3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
		float fy = input3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];
		
		//get the destination position
		float x2 = (float)(w_i) + fx/2.0f; //the intermediate position
		float y2 = (float)(h_i) + fy/2.0f;

		//Guarrantee that the center position is in-border.
		if(x2 >= 0.0f && y2 >=0.0f && x2 <=  (float) (w -1) && y2  <= (float) (h-1)){
			int ix2_L = (int)(x2);
			int iy2_T = (int)(y2);
   
			float alpha = x2 - (int)(x2);
			float beta =  y2 - (int)(y2);
			
			int m;
			int n;
			// we interpolate 4 pixels, should we change the sigma_d ?
			for( m = -1; m <= 2; m ++){
 				for(n = -1; n <= 2; n ++){
 					int patch2_m = min(max(0, m + iy2_T), h-1);
					int patch2_n = min(max(0, n + ix2_L), w-1);
					
//					float g_d = expf( - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                    float g_d = (1.0f - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                    //        g_d = g_d * g_d;
					float f_w = flow_weights[batch_i * flow_weights_b_stride + 0+ h_i * flow_weights_h_stride + w_i]  ;
					
					float gradoutput_data_value = gradoutput[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n];
// 					if(output[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n] < 0)
//                        printf("Error g_d ==> %f \n",output[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n] );
                    if(output[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n] < threshhold)
                    {
                        //printf("pixelweigths gpu backward, under threshhold ==> %f\n",
                         //   output[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n]);
                              continue;//to  skip its gradients
					}
					//flow1_weights gradients
					atomicAdd(&gradflow_weights[batch_i * flow_weights_b_stride + 0+ h_i * flow_weights_h_stride + w_i],
						gradoutput_data_value * g_d * g_d);
						
					//flow gradients
					atomicAdd(& gradinput3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i],
							- gradoutput_data_value * f_w * g_d * (n - alpha) / ( sigma_d * sigma_d) * 2.0f);
					atomicAdd(& gradinput3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i] ,
							- gradoutput_data_value * f_w *  g_d *(m - beta)  / (  sigma_d * sigma_d) * 2.0f);
 				}												
			}											
		}
	}	
	return ;

}
int PixelWeightLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    			const int batch,

		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		
		at::Tensor&  input3, const at::Tensor&  flow_weights,	at::Tensor&  output,

		at::Tensor&  gradoutput,  at::Tensor&  gradinput3, at::Tensor&  gradflow_weights,
        float threshhold,
		float	sigma_d,     float tao_r , float  Prowindow

		)
{
    int error = 1 ;

	dim3 grid;
	dim3 block;


	//blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);

//    cudaMemset((void*)gradinput1, 0, input1_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput2, 0, input2_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput3, 0, input3_b_stride * batch * sizeof(float));

	PixelWeightLayer_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,
			//input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,


			input3,  flow_weights, output,
			gradoutput, gradinput3, gradflow_weights,
            threshhold,
            sigma_d,      tao_r ,   Prowindow

			);

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;		
}



//forward path of our layer
__global__ void ReliableWeightLayer_gpu_forward_kernelfunc(
		const int nElement,
		const int w, 		const int h,

		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor&  input3, 	at::Tensor&  output,
		float	sigma_d,     float tao_r , float  Prowindow
		)
{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	//const int off = batch_i * input1_b_stride;


	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {
			//read the opticalflow
		float fx = input3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
		float fy = input3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];

		//get the destination position
		float x2 = (float)(w_i) + fx/2.0f; //the intermediate position
		float y2 = (float)(h_i) + fy/2.0f;
		//G	uarrantee that the center position is in-border.
		if(x2 >= 0.0f && y2 >=0.0f && x2 <=  (float) (w -1) && y2  <= (float) (h-1)){
			int ix2_L = (int)(x2);
			int iy2_T = (int)(y2);
  			
			float alpha = x2 - (int)(x2);
			float beta =  y2  - (int)(y2);
			
			int m;
			int n;
			// we interpolate 4 pixels, should we change the sigma_d ?
			for( m = -1; m <= 2; m ++){										
				for(n = -1; n <= 2; n ++){				
					int patch2_m = min(max(0, m + iy2_T), h-1);
					int patch2_n = min(max(0, n + ix2_L), w-1);					
//					float g_d =  expf( - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                    float g_d = (1.0f - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                            g_d = g_d * g_d;
  					atomicAdd(&output[batch_i * output_b_stride + 0 +  patch2_m * output_h_stride + patch2_n], g_d);
 				}												
			}											
		}else{
			;
 		}
	}
	return ;

}
int ReliableWeightLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, const int batch,

		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor&  input3,  	at::Tensor&  output,
		float	sigma_d,     float tao_r , float  Prowindow
		)
{
	int error = 1 ;

	dim3 grid;
	dim3 block;


	// blockthread = 128;
	// the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	// the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	//extract the data of CudaTensor and use kernel to calculate.
	ReliableWeightLayer_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,
			//input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			//flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,

			input3,  output,
			sigma_d,  tao_r,   Prowindow
			);

	//			THCudaCheck(cudaGetLastError());
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;
}



__global__ void ReliableWeightLayer_gpu_backward_kernelfunc(
		const int nElement, 	   const int w, 		const int h,
		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor&  input3,at::Tensor&  output,
		at::Tensor&  gradoutput, at::Tensor&  gradinput3,  
        float threshhold,
		float	sigma_d,     float tao_r , float  Prowindow
		)
		{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	//const int off  = batch_i * input1_b_stride;

	//    __syncthreads();

	if(withinXbounds && withinYbounds) 
	{
				//read the opticalflow
		float fx = input3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
		float fy = input3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];
		
		//get the destination position
		float x2 = (float)(w_i) + fx/2.0f; //the intermediate position
		float y2 = (float)(h_i) + fy/2.0f;
		//Guarrantee that the center position is in-border.
		if(x2 >= 0.0f && y2 >=0.0f && x2 <=  (float) (w -1) && y2  <= (float) (h-1)){
			int ix2_L = (int)(x2);
			int iy2_T = (int)(y2);

			float alpha = x2 - (int)(x2);
			float beta =  y2  - (int)(y2);
			
			int m;
			int n;
			// we interpolate 4 pixels, should we change the sigma_d ?
			for( m = -1; m <= 2; m ++){
				for(n = -1; n <= 2; n ++){
					
					int patch2_m = min(max(0, m + iy2_T), h-1);
					int patch2_n = min(max(0, n + ix2_L), w-1);
					
//					float g_d =  expf( - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                    float g_d = (1.0f - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
//                          g_d = g_d * g_d;
					float gradoutput_data_value = gradoutput[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n];
//					if(output[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n] < 0)
 //                           printf("Error g_d ==> %f \n",output[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n] );
                             if(output[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n] < threshhold)
                            {
                                //printf("Reliable gpu backward, under threshhold ==> %f\n",
                                 //   output[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n]);
                              continue;//to  skip its gradients
					        }
					//flow gradients
					atomicAdd( & gradinput3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i],
							- gradoutput_data_value *  g_d * (n - alpha) / ( sigma_d * sigma_d) * 2.0f);
					atomicAdd( & gradinput3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i],
							- gradoutput_data_value *   g_d *(m - beta)  / (  sigma_d * sigma_d) * 2.0f);
				}												
			}											
		}
	}	
	return ;

}

int ReliableWeightLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    			const int batch,

		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		
		at::Tensor&  input3,   at::Tensor&  output,
 
		at::Tensor&  gradoutput,  at::Tensor&  gradinput3,
        float threshhold,
		float	sigma_d,     float tao_r , float  Prowindow

		)
{
	int error = 1 ;

	dim3 grid;
	dim3 block;


	//blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);

//    cudaMemset((void*)gradinput1, 0, input1_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput2, 0, input2_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput3, 0, input3_b_stride * batch * sizeof(float));

	ReliableWeightLayer_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,
			//input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			//flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,


			input3, output,
			gradoutput, gradinput3,
			threshhold,
			sigma_d,      tao_r ,   Prowindow

			);

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;		
}
//#ifdef __cplusplus
//	}
//#endif
