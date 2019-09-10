#include <THC.h>
//#include <THCGeneral.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include "my_lib_kernel.h" //MUST be included

//an important state pointer for the management
extern THCState* state;


int SeparableConvFlowLayer_gpu_forward(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  input3,
		//at::Tensor&  output,
		at::Tensor&  flow_output

		)
		{
	int error = 1 ;
    //int point  =0 ;printf("debug point  %d\n", point++ );

	int channel = input1.size( 1);
	if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size(0) != batch) return error;
	if(input2.size(1) != input2.size(1)) return error;
    //printf("debug point  %d\n", point++ );

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h - input2.size(1) + 1) return error;// to add some checkpoint
	if(input2.size(3) != w - input2.size(1) + 1) return error;
	

	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);

    //int output_b_stride = output.stride(0);
	//int output_c_stride = output.stride(1);
	//int output_h_stride = output.stride(2);
	//int output_w_stride = output.stride(3);
	
    int flow_output_b_stride = flow_output.stride(0);
	int flow_output_c_stride = flow_output.stride(1);
	int flow_output_h_stride = flow_output.stride(2);
	int flow_output_w_stride = flow_output.stride(3);	
    //printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
    if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
   // if(output_w_stride !=1) return error;
	if(flow_output_w_stride !=1) return error;


	if(input2_b_stride != input3_b_stride) return error;
	if(input2_c_stride != input3_c_stride) return error;
    //printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	int	nElement = 0;//UNUSED  0;//UNUSED  THCudaTensor_nElement(state, flow_output);


	error = SeparableConvFlowLayer_gpu_forward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement,w,h,channel,batch,  input2.size(1),

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
		//	output_b_stride,output_c_stride,output_h_stride,output_w_stride,
			flow_output_b_stride,flow_output_c_stride,flow_output_h_stride,flow_output_w_stride,



			input1,
			input2,
			input3,
			//output ,
			flow_output 
			
			);
	  if (error) {AT_ERROR("CUDA call failed");}
    //printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);

	return error;

		}
int SeparableConvFlowLayer_gpu_backward(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  input3,
		at::Tensor&  gradflow_output,
		at::Tensor&  gradinput1,
		at::Tensor&  gradinput2,
		at::Tensor&  gradinput3
		)
		{


    int error = 1 ;
	int channel = input1.size( 1);
	if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != input2.size(1)) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h - input2.size(1) + 1) return error;// to add some checkpoint
	if(input2.size(3) != w - input2.size(1) + 1) return error;


	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);

    //int output_b_stride = gradoutput.stride(0);
	//int output_c_stride = gradoutput.stride(1);
	//int output_h_stride = gradoutput.stride(2);
	//int output_w_stride = gradoutput.stride(3);
	
    int flow_output_b_stride = gradflow_output.stride(0);
	int flow_output_c_stride = gradflow_output.stride(1);
	int flow_output_h_stride = gradflow_output.stride(2);
	int flow_output_w_stride = gradflow_output.stride(3);		

//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
  //  if(output_w_stride !=1) return error;
	if(flow_output_w_stride !=1) return error;

    if(input1_b_stride != gradinput1.stride(0)) return error;
	if(input2_b_stride != gradinput2.stride(0)) return error;
	if(input1_c_stride != gradinput1.stride(1)) return error;
	if(input2_c_stride != gradinput2.stride(1)) return error;
	if(input3_c_stride != gradinput3.stride(1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);

	int	nElement = 0;//UNUSED  0;//UNUSED  THCudaTensor_nElement(state, gradflow_output);

	error  = SeparableConvFlowLayer_gpu_backward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement, //to let the nummous
			w,h,channel,batch,  input2.size(1),

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
		//	output_b_stride,output_c_stride,output_h_stride,output_w_stride,
			flow_output_b_stride,flow_output_c_stride,flow_output_h_stride,flow_output_w_stride,

			input1,
			input2,
			input3,
			gradflow_output,
			gradinput1,
			gradinput2,
			gradinput3
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;
}


int SeparableConvLayer_gpu_forward(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  input3,
		at::Tensor&  output

		)
		{
	int error = 1 ;

	int channel = input1.size( 1);
	if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != input2.size(1)) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h - input2.size(1) + 1) return error;// to add some checkpoint
	if(input2.size(3) != w - input2.size(1) + 1) return error;


	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);

    int output_b_stride = output.stride(0);
	int output_c_stride = output.stride(1);
	int output_h_stride = output.stride(2);
	int output_w_stride = output.stride(3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
    if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
    if(output_w_stride !=1) return error;

	if(input2_b_stride != input3_b_stride) return error;
	if(input2_c_stride != input3_c_stride) return error;


	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, output);


	error = SeparableConvLayer_gpu_forward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement,w,h,channel,batch,  input2.size(1),

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,



			input1,
			input2,
			input3,
			output);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;

		}
int SeparableConvLayer_gpu_backward(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  input3,
		at::Tensor&  gradoutput,
		at::Tensor&  gradinput1,
		at::Tensor&  gradinput2,
		at::Tensor&  gradinput3
		)
		{


    int error = 1 ;
	int channel = input1.size( 1);
	if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != input2.size(1)) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h - input2.size(1) + 1) return error;// to add some checkpoint
	if(input2.size(3) != w - input2.size(1) + 1) return error;


	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);

    int output_b_stride = gradoutput.stride(0);
	int output_c_stride = gradoutput.stride(1);
	int output_h_stride = gradoutput.stride(2);
	int output_w_stride = gradoutput.stride(3);

//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
    if(output_w_stride !=1) return error;

    if(input1_b_stride != gradinput1.stride(0)) return error;
	if(input2_b_stride != gradinput2.stride(0)) return error;
	if(input1_c_stride != gradinput1.stride(1)) return error;
	if(input2_c_stride != gradinput2.stride(1)) return error;
	if(input3_c_stride != gradinput3.stride(1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, gradoutput);

	error  = SeparableConvLayer_gpu_backward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement, //to let the nummous
			w,h,channel,batch,  input2.size(1),

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,

			input1,
			input2,
			input3,
			gradoutput,
			gradinput1,
			gradinput2,
			gradinput3
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;
}

int InterpolationLayer_gpu_forward(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  output
		)
		{
	int error = 1 ;

	int channel = input1.size( 1);
	if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != 2) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h) return error;// to add some checkpoint
	if(input2.size(3) != w) return error;

	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);
	//TODO: do we need to assert the w_stride to be 1
	//if(w_stride !=1) return error;
	if(input1_b_stride != output.stride(0)) return error;
	if(input1_c_stride != output.stride(1)) return error;

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, output);

	error =InterpolationLayer_gpu_forward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement,w,h,channel,batch,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,

			input1,
			input2,
			output);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;

}


int InterpolationLayer_gpu_backward(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  gradoutput,
		at::Tensor&  gradinput1,
		at::Tensor&  gradinput2
		)
    {
	int error = 1 ;
	int channel = input1.size( 1);
	if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != 2) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h) return error;// to add some checkpoint
	if(input2.size(3) != w) return error;

	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);
	//TODO: do we need to assert the w_stride to be 1
	//if(w_stride !=1) return error;
	if(input1_b_stride != gradinput1.stride(0)) return error;
	if(input2_b_stride != gradinput2.stride(0)) return error;
	if(input1_c_stride != gradinput1.stride(1)) return error;
	if(input2_c_stride != gradinput2.stride(1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, gradoutput);

	error  = InterpolationLayer_gpu_backward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement, //to let the nummous
			w,h,channel,batch,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,

			input1,
			input2,
			gradoutput,
			gradinput1,
			gradinput2
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;

}

int InterpolationChLayer_gpu_forward(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  output
		)
		{
	int error = 1 ;

	int channel = input1.size( 1);
//	if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != 2) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h) return error;// to add some checkpoint
	if(input2.size(3) != w) return error;

	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);
	//TODO: do we need to assert the w_stride to be 1
	//if(w_stride !=1) return error;
	if(input1_b_stride != output.stride(0)) return error;
	if(input1_c_stride != output.stride(1)) return error;

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, output);

	error =InterpolationLayer_gpu_forward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement,w,h,channel,batch,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,

			input1,
			input2,
			output);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;

}


int InterpolationChLayer_gpu_backward(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  gradoutput,
		at::Tensor&  gradinput1,
		at::Tensor&  gradinput2
		)
    {
	int error = 1 ;
	int channel = input1.size( 1);
//	if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != 2) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h) return error;// to add some checkpoint
	if(input2.size(3) != w) return error;


	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);
	//TODO: do we need to assert the w_stride to be 1
	//if(w_stride !=1) return error;
	if(input1_b_stride != gradinput1.stride(0)) return error;
	if(input2_b_stride != gradinput2.stride(0)) return error;
	if(input1_c_stride != gradinput1.stride(1)) return error;
	if(input2_c_stride != gradinput2.stride(1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, gradoutput);

	error  = InterpolationLayer_gpu_backward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement, //to let the nummous
			w,h,channel,batch,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,

			input1,
			input2,
			gradoutput,
			gradinput1,
			gradinput2
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;

}

int FilterInterpolationLayer_gpu_forward(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  input3,
		at::Tensor&  output

		)
		{
	int error = 1 ;

	int channel = input1.size( 1);
	//if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != 2) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h) return error;// to add some checkpoint
	if(input2.size(3) != w) return error;

    int filter_size2 = input3.size( 1);
    int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));


	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
    if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
	if(input1_b_stride != output.stride(0)) return error;
	if(input1_c_stride != output.stride(1)) return error;

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, output);


	error = FilterInterpolationLayer_gpu_forward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement,w,h,channel,batch, filter_size,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,


			input1,
			input2,
			input3,
			output);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;

		}
int FilterInterpolationLayer_gpu_backward(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  input3,
		at::Tensor&  gradoutput,
		at::Tensor&  gradinput1,
		at::Tensor&  gradinput2,
		at::Tensor&  gradinput3
		)
		{


    int error = 1 ;
	int channel = input1.size( 1);
	//if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input2.size(1) != 2) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h) return error;// to add some checkpoint
	if(input2.size(3) != w) return error;


    int filter_size2 = input3.size( 1);
    int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));

	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
    if(input1_b_stride != gradinput1.stride(0)) return error;
	if(input2_b_stride != gradinput2.stride(0)) return error;
	if(input1_c_stride != gradinput1.stride(1)) return error;
	if(input2_c_stride != gradinput2.stride(1)) return error;
	if(input3_c_stride != gradinput3.stride(1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, gradoutput);

	error  = FilterInterpolationLayer_gpu_backward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement, //to let the nummous
			w,h,channel,batch, filter_size,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,

			input1,
			input2,
			input3,
			gradoutput,
			gradinput1,
			gradinput2,
			gradinput3
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;
}


int FlowProjectionLayer_gpu_forward(
		at::Tensor&  input1,
		at::Tensor&  count,
		at::Tensor&  output,
		int fillhole
		)
{

	int error = 1 ;

	int channel = input1.size( 1);
	if(channel!= 2) return error;
	int batch = input1.size(0);

	int h = input1.size(2);
	int w = input1.size(3);

	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int count_b_stride = count.stride(0);
	int count_c_stride = count.stride(1);
	int count_h_stride = count.stride(2);
	int count_w_stride = count.stride(3);
	//TODO: do we need to assert the w_stride to be 1
	//if(w_stride !=1) return error;
	if(input1_b_stride != output.stride(0)) return error;
	if(input1_c_stride != output.stride(1)) return error;

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, output);
//    printf("In gpu forward\n");
	error = FlowProjection_gpu_forward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement,w,h,channel,batch,fillhole,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,

			input1,
			count,
			output);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;

}

int FlowProjectionLayer_gpu_backward(
		at::Tensor&  input1,
        at::Tensor&  count,
		at::Tensor&  gradoutput,
		at::Tensor&  gradinput1
		)
{
	int error = 1 ;
	int channel = input1.size( 1);
	if(channel!=2) return error;
	int batch = input1.size(0);
	if(count.size(0) != batch) return error;
	if(count.size(1) != 1) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(count.size(2) != h) return error;// to add some checkpoint
	if(count.size(3) != w) return error;

	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int count_b_stride = count.stride(0);
	int count_c_stride = count.stride(1);
	int count_h_stride = count.stride(2);
	int count_w_stride = count.stride(3);
	//TODO: do we need to assert the w_stride to be 1
	//if(w_stride !=1) return error;
	if(input1_b_stride != gradinput1.stride(0)) return error;
	if(input1_c_stride != gradinput1.stride(1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);
//    printf("GPU backward: %d,%d,%d,%d\n", count_b_stride,count_c_stride,count_h_stride,count_w_stride);

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, gradoutput);

	error  = FlowProjection_gpu_backward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement, //to let the nummous
			w,h,channel,batch,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,

			input1,
			count,
			gradoutput,
			gradinput1
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;

}
int WeightedFlowProjectionLayer_gpu_forward(
		at::Tensor&  input1,
		at::Tensor&  input2,
		at::Tensor&  input3,
		at::Tensor&  count,
		at::Tensor&  weight,
		at::Tensor&  output,
		int fillhole,
		float threshold
		)
{

	int error = 1 ;

	int channel = input1.size( 1);
	if(channel!= 2) return error;
	if(input2.size( 1) !=3) return error;
	if(input3.size( 1) !=3) return error;

	int batch = input1.size(0);

	int h = input1.size(2);
	int w = input1.size(3);

	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);

	int count_b_stride = count.stride(0);
	int count_c_stride = count.stride(1);
	int count_h_stride = count.stride(2);
	int count_w_stride = count.stride(3);

	int weight_b_stride = weight.stride(0);
	int weight_c_stride = weight.stride(1);
	int weight_h_stride = weight.stride(2);
	int weight_w_stride = weight.stride(3);
	//TODO: do we need to assert the w_stride to be 1
	//if(w_stride !=1) return error;
	if(input1_b_stride != output.stride(0)) return error;
	if(input1_c_stride != output.stride(1)) return error;

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, output);
//    printf("In gpu forward\n");
	error = WeightedFlowProjection_gpu_forward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement,w,h,channel,batch,fillhole, threshold,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
            input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,
			weight_b_stride,weight_c_stride,weight_h_stride,weight_w_stride,

			input1,
			input2,
			input3,
			count,
			weight,
			output);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;

}

int WeightedFlowProjectionLayer_gpu_backward(
		at::Tensor&  input1,
        at::Tensor&  input2,
		THCudaTensor  * input3,
        at::Tensor&  count,
        at::Tensor&  weight,
		at::Tensor&  gradoutput,
		at::Tensor&  gradinput1,
		float threshhold
		)
{
	int error = 1 ;
	int channel = input1.size( 1);
	if(channel!=2) return error;
	if(input2.size( 1) !=3) return error;
	if(input3.size( 1) !=3) return error;
	int batch = input1.size(0);
	if(count.size( 0) != batch) return error;
	if(count.size(1) != 1) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(count.size(2) != h) return error;// to add some checkpoint
	if(count.size(3) != w) return error;

	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);

	int count_b_stride = count.stride(0);
	int count_c_stride = count.stride(1);
	int count_h_stride = count.stride(2);
	int count_w_stride = count.stride(3);

	int weight_b_stride = weight.stride(0);
	int weight_c_stride = weight.stride(1);
	int weight_h_stride = weight.stride(2);
	int weight_w_stride = weight.stride(3);
	//TODO: do we need to assert the w_stride to be 1
	//if(w_stride !=1) return error;
	if(input1_b_stride != gradinput1.stride(0)) return error;
	if(input1_c_stride != gradinput1.stride(1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);
//    printf("GPU backward: %d,%d,%d,%d\n", count_b_stride,count_c_stride,count_h_stride,count_w_stride);

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, gradoutput);

	error  = WeightedFlowProjection_gpu_backward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement, //to let the nummous
			w,h,channel,batch, threshhold,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
            input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,
			weight_b_stride,weight_c_stride,weight_h_stride,weight_w_stride,

			input1,
            input2,
			input3,
			count,
            weight,
			gradoutput,
			gradinput1
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;

}


int WeightLayer_gpu_forward(
		at::Tensor&  input1, 		at::Tensor&  input2, 		at::Tensor&  input3,
//		at::Tensor&  flow1_grad,
    at::Tensor&  output,
		float  lambda_e, float	lambda_v, float Nw
		)
{
	int error = 1 ;
	//printf("Entering WeightLayer_gpu_forward\n");
	if(Nw != 3.0f) {printf("Nw must be odd number 3\n"); return error;}

	int channel = input1.size( 1);
	if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;

	//printf("Entering WeightLayer_gpu_forward\n");
	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h) return error;// to add some checkpoint
	if(input2.size(3) != w) return error;

	//printf("Entering WeightLayer_gpu_forward\n");
//    int filter_size2 = input3.size( 1);
  //  int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(input3.size( 1) != 2){ printf("the flow channel should be 2\n"); return error;}
//	if(flow1_grad.size(1) != 1){ printf("the flow1_grad channel should be 1\n"); return error;}
	if(output.size(1) != 1) {printf("the flow weight channel should be 1\n"); return error;}


	//printf("Entering WeightLayer_gpu_forward\n");
	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);
	
//	int flow1_grad_b_stride = flow1_grad.stride(0);
//	int flow1_grad_c_stride = flow1_grad.stride(1);
//	int flow1_grad_h_stride = flow1_grad.stride(2);
//	int flow1_grad_w_stride = flow1_grad.stride(3);
	
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);
	int output_b_stride = output.stride(0);
	int output_c_stride = output.stride(1);
	int output_h_stride = output.stride(2);
	int output_w_stride = output.stride(3);	

	//TODO: do we need to assert the w_stride to be 1
    if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
//    if(flow1_grad_w_stride != 1) return error;
    if(output_w_stride !=1 )return error;
	//printf("Entering WeightLayer_gpu_forward\n");
//	if(input1_b_stride != output.stride(0)) return error;
//	if(input1_c_stride != output.stride(1)) return error;

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, output);


	error = WeightLayer_gpu_forward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement,w,h,channel,batch,  

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
//			flow1_grad_b_stride,flow1_grad_c_stride,flow1_grad_h_stride,flow1_grad_w_stride,
			output_b_stride,  output_c_stride,   output_h_stride,   output_w_stride,

			input1,
			input2,
			input3,
//			flow1_grad,
			output,
			
			lambda_e,  lambda_v,   Nw
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;
	
}
int WeightLayer_gpu_backward(
		at::Tensor&  input1, 		at::Tensor&  input2, 		at::Tensor&  input3,
//		at::Tensor&  flow1_grad,
    at::Tensor&  output,
		at::Tensor&  gradoutput, 
		at::Tensor&  gradinput1,		at::Tensor&  gradinput2, 		at::Tensor&  gradinput3,
//		at::Tensor&   gradflow1_grad,
		float  lambda_e, float	lambda_v, float Nw
		)
{
		
    int error = 1 ;
	if(Nw != 3.0f) {printf("Nw must be odd number 3\n"); return error;}

	int channel = input1.size( 1);
	if(channel!=3) return error;
	int batch = input1.size(0);
	if(input2.size( 0) != batch) return error;
	if(input3.size(1) != 2) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	if(input2.size(2) != h) return error;// to add some checkpoint
	if(input2.size(3) != w) return error;


   // int filter_size2 = input3.size( 1);
    //int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(input3.size( 1) != 2) {printf("the flow channel should be 2\n"); return error;}
//	if(flow1_grad.size(1) != 1) {printf("the flow1_grad channel should be 1\n"); return error;}
	if(output.size(1) != 1) {printf("the flow weight channel should be 1\n"); return error;}

	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	int input2_b_stride = input2.stride(0);
	int input2_c_stride = input2.stride(1);
	int input2_h_stride = input2.stride(2);
	int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);

	
//	int flow1_grad_b_stride = flow1_grad.stride(0);
//	int flow1_grad_c_stride = flow1_grad.stride(1);
//	int flow1_grad_h_stride = flow1_grad.stride(2);
//	int flow1_grad_w_stride = flow1_grad.stride(3);
	
	int output_b_stride = output.stride(0);
	int output_c_stride = output.stride(1);
	int output_h_stride = output.stride(2);
	int output_w_stride = output.stride(3);	

	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
    if(input1_b_stride != gradinput1.stride(0)) return error;
	if(input2_b_stride != gradinput2.stride(0)) return error;
	if(input1_c_stride != gradinput1.stride(1)) return error;
	if(input2_c_stride != gradinput2.stride(1)) return error;
	if(input3_c_stride != gradinput3.stride(1)) return error;

    //printf("WeightLayer_gpu_backward GPU backward: input1 %d,%d,%d,%d\n",
    //    input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);
    //printf("WeightLayer_gpu_backward GPU backward: flow1_grad %d,%d,%d,%d\n",
	//		flow1_grad_b_stride,flow1_grad_c_stride,flow1_grad_h_stride,flow1_grad_w_stride);

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, gradoutput);

	error  = WeightLayer_gpu_backward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement, //to let the nummous
			w,h,channel,batch,  

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
//			flow1_grad_b_stride,flow1_grad_c_stride,flow1_grad_h_stride,flow1_grad_w_stride,
			output_b_stride,  output_c_stride,   output_h_stride,   output_w_stride,

			input1,
			input2,
			input3,
//			flow1_grad,
			output,
			

			gradoutput,
			gradinput1,
			gradinput2,
			gradinput3,
//			gradflow1_grad,
			lambda_e,  lambda_v,   Nw
 
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;
}
		
int PixelValueLayer_gpu_forward(
		at::Tensor&  input1,  		at::Tensor&  input3, 	at::Tensor&  flow_weights, 		at::Tensor&  output,
		float sigma_d,    float tao_r ,  float Prowindow
		)
{
	int error = 1 ;
	if(Prowindow != 2.0f) {printf("Prowindow must be odd number 2\n"); return error;}

	int channel = input1.size( 1);
	if(channel!=3) return error;
	int batch = input1.size(0);
	//if(input2.size( 0) != batch) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	//if(input2.size(2) != h) return error;// to add some checkpoint
	//if(input2.size(3) != w) return error;

//    int filter_size2 = input3.size( 1);
  //  int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(input3.size( 1) != 2) {printf("the flow channel should be 2\n"); return error;}
	if(flow_weights(1) != 1){ printf("the flow_weights channel should be 1\n"); return error;}
	if(output.size(1) != 3) {printf("the image channel should be 3\n"); return error;}


	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	///int input2_b_stride = input2.stride(0);
	//int input2_c_stride = input2.stride(1);
	//int input2_h_stride = input2.stride(2);
	//int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);
	
	int flow_weights_b_stride = flow_weights.stride(0);
	int flow_weights_c_stride = flow_weights.stride(1);
	int flow_weights_h_stride = flow_weights.stride(2);
	int flow_weights_w_stride = flow_weights.stride(3);	
	
	int output_b_stride = output.stride(0);
	int output_c_stride = output.stride(1);
	int output_h_stride = output.stride(2);
	int output_w_stride = output.stride(3);	
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
    if(input1_w_stride !=1) return error;
	//if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
	if(flow_weights_w_stride !=1) return error;
	if(input1_b_stride != output.stride(0)) return error;
	if(input1_c_stride != output.stride(1)) return error;

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, output);


	error = PixelValueLayer_gpu_forward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement,w,h,channel,batch,  

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,

			input1,
			//input2,
			input3,
			flow_weights,
			output,
			
			sigma_d,      tao_r ,   Prowindow
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;		
}
int PixelValueLayer_gpu_backward(
		at::Tensor&  input1,  		at::Tensor&  input3, 	at::Tensor&  flow_weights, 		 
		at::Tensor&  gradoutput, 
		at::Tensor&  gradinput1,		at::Tensor&  gradinput3, at::Tensor&  gradflow_weights,
		float sigma_d,    float tao_r ,  float Prowindow
		)
{
	int error = 1 ;
	if(Prowindow != 2.0f) {printf("Prowindow must be odd number 2\n"); return error;}

	int channel = input1.size( 1);
	if(channel!=3) return error;
	int batch = input1.size(0);
	//if(input2.size( 0) != batch) return error;
	//if(input2.size(1) != 2) return error;

	int h = input1.size(2);
	int w = input1.size(3);
	//if(input2.size(2) != h) return error;// to add some checkpoint
	//if(input2.size(3) != w) return error;


   // int filter_size2 = input3.size( 1);
    //int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(input3.size( 1) != 2) {printf("the flow channel should be 2\n"); return error;}
	if(flow_weights(1) != 1) {printf("the flow_weights channel should be 1\n"); return error;}
	//if(output.size(1) != 1){ printf("the flow weight channel should be 1\n"); return error;}

	int input1_b_stride = input1.stride(0);
	int input1_c_stride = input1.stride(1);
	int input1_h_stride = input1.stride(2);
	int input1_w_stride = input1.stride(3);

	//int input2_b_stride = input2.stride(0);
	//int input2_c_stride = input2.stride(1);
	//int input2_h_stride = input2.stride(2);
	//int input2_w_stride = input2.stride(3);
     
    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	int flow_weights_b_stride = flow_weights.stride(0);
	int flow_weights_c_stride = flow_weights.stride(1);
	int flow_weights_h_stride = flow_weights.stride(2);
	int flow_weights_w_stride = flow_weights.stride(3);	
	
	int output_b_stride = gradoutput.stride(0);
	int output_c_stride = gradoutput.stride(1);
	int output_h_stride = gradoutput.stride(2);
	int output_w_stride = gradoutput.stride(3);		

	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	//if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
    if(input1_b_stride != gradinput1.stride(0)) return error;
	//if(input2_b_stride != gradinput2.stride(0)) return error;
	if(input1_c_stride != gradinput1.stride(1)) return error;
	//if(input2_c_stride != gradinput2.stride(1)) return error;
	if(input3_c_stride != gradinput3.stride(1)) return error;
	if(flow_weights_b_stride != gradflow_weights.stride(0)) return error;

//    printf("PixelValueLayer_gpu_backward GPU backward:  input1 %d,%d,%d,%d\n",
//        input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);
//    printf("PixelValueLayer_gpu_backward GPU backward: flow weights %d,%d,%d,%d\n",
//        gradflow_weights.stride(0),gradflow_weights.stride(1),
//        gradflow_weights.stride(2),gradflow_weights.stride(3));

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, gradoutput);

	error  = PixelValueLayer_gpu_backward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement, //to let the nummous
			w,h,channel,batch,  

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,


			input1,
			//input2,
			input3,
			flow_weights,
			//output,

			gradoutput,
			gradinput1,
			//gradinput2,
			gradinput3,
			gradflow_weights,
			sigma_d,      tao_r ,   Prowindow
 
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;		
}
int PixelWeightLayer_gpu_forward(
		at::Tensor&  input3, 	at::Tensor&  flow_weights, 		at::Tensor&  output,
		float sigma_d,    float tao_r ,  float Prowindow
		)
{
	int error = 1 ;
	if(Prowindow != 2.0f) {printf("Prowindow must be odd number 2\n"); return error;}

	//int channel = input1.size( 1);
	//if(channel!=3) return error;
	int batch = input3.size(0);
	//if(input2.size( 0) != batch) return error;

	int h = input3.size(2);
	int w = input3.size(3);
	//if(input2.size(2) != h) return error;// to add some checkpoint
	//if(input2.size(3) != w) return error;

//    int filter_size2 = input3.size( 1);
  //  int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(input3.size( 1) != 2) {printf("the flow channel should be 2\n"); return error;}
	if(flow_weights(1) != 1){ printf("the flow_weights channel should be 1\n"); return error;}
	if(output.size(1) != 1) {printf("the flow weight channel should be 1\n"); return error;}


	//int input1_b_stride = input1.stride(0);
	//int input1_c_stride = input1.stride(1);
	//int input1_h_stride = input1.stride(2);
	//int input1_w_stride = input1.stride(3);

	///int input2_b_stride = input2.stride(0);
	//int input2_c_stride = input2.stride(1);
	//int input2_h_stride = input2.stride(2);
	//int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);
	
	int flow_weights_b_stride = flow_weights.stride(0);
	int flow_weights_c_stride = flow_weights.stride(1);
	int flow_weights_h_stride = flow_weights.stride(2);
	int flow_weights_w_stride = flow_weights.stride(3);	
	
	int output_b_stride = output.stride(0);
	int output_c_stride = output.stride(1);
	int output_h_stride = output.stride(2);
	int output_w_stride = output.stride(3);	
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
    //if(input1_w_stride !=1) return error;
	//if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
	if(flow_weights_w_stride !=1) return error;
	///if(input1_b_stride != output.stride(0)) return error;
	//if(input1_c_stride != output.stride(1)) return error;

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, output);


	error = PixelWeightLayer_gpu_forward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement,w,h, batch,  

			//input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,

			//input1,
			//input2,
			input3,
			flow_weights,
			output,
			
			sigma_d,      tao_r ,   Prowindow
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;				
}
int PixelWeightLayer_gpu_backward(
		at::Tensor&  input3, 	at::Tensor&  flow_weights, 	at::Tensor&  output,
		at::Tensor&  gradoutput, 
		at::Tensor&  gradinput3, at::Tensor&  gradflow_weights,
		float threshhold,
		float sigma_d,    float tao_r ,  float Prowindow
		)
{
	int error = 1 ;
	if(Prowindow != 2.0f) {printf("Prowindow must be odd number 2\n"); return error;}

	//int channel = input1.size( 1);
	//if(channel!=3) return error;
	int batch = input3.size(0);
	//if(input2.size( 0) != batch) return error;
	//if(input2.size(1) != 2) return error;

	int h = input3.size(2);
	int w = input3.size(3);
	//if(input2.size(2) != h) return error;// to add some checkpoint
	//if(input2.size(3) != w) return error;


   // int filter_size2 = input3.size( 1);
    //int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(input3.size( 1) != 2) {printf("the flow channel should be 2\n"); return error;}
	if(flow_weights(1) != 1) {printf("the flow_weights channel should be 1\n"); return error;}
	//if(output.size(1) != 1){ printf("the flow weight channel should be 1\n"); return error;}

	//int input1_b_stride = input1.stride(0);
	//int input1_c_stride = input1.stride(1);
	//int input1_h_stride = input1.stride(2);
	//int input1_w_stride = input1.stride(3);

	//int input2_b_stride = input2.stride(0);
	//int input2_c_stride = input2.stride(1);
	//int input2_h_stride = input2.stride(2);
	//int input2_w_stride = input2.stride(3);
     
    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	int flow_weights_b_stride = flow_weights.stride(0);
	int flow_weights_c_stride = flow_weights.stride(1);
	int flow_weights_h_stride = flow_weights.stride(2);
	int flow_weights_w_stride = flow_weights.stride(3);	
	
	int output_b_stride = gradoutput.stride(0);
	int output_c_stride = gradoutput.stride(1);
	int output_h_stride = gradoutput.stride(2);
	int output_w_stride = gradoutput.stride(3);		

	//TODO: do we need to assert the w_stride to be 1
	//if(input1_w_stride !=1) return error;
	//if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
    //if(input1_b_stride != gradinput1.stride(0)) return error;
	//if(input2_b_stride != gradinput2.stride(0)) return error;
	//if(input1_c_stride != gradinput1.stride(1)) return error;
	//if(input2_c_stride != gradinput2.stride(1)) return error;
	if(input3_c_stride != gradinput3.stride(1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, gradoutput);

	error  = PixelWeightLayer_gpu_backward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement, //to let the nummous
			w,h, batch,  

			//input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,


			//input1,
			//input2,
			input3,
			flow_weights,
			output,

			gradoutput,
			//gradinput1,
			//gradinput2,
			gradinput3,
			gradflow_weights,
			threshhold,
			sigma_d,      tao_r ,   Prowindow
 
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;			
}
		
//int ReliableValueLayer_gpu_forward(
//		at::Tensor&  input3, 	at::Tensor&  flow_weights, 		at::Tensor&  output,
//		float sigma_d,    float tao_r ,  float Prowindow
//		)
//{
//return 0;
//}
//int ReliableValueLayer_gpu_backward(
//		at::Tensor&  input3, 	at::Tensor&  flow_weights, 		at::Tensor&  output,
//		at::Tensor&  gradinput3, at::Tensor&  gradflow_weights,
//		float sigma_d,    float tao_r ,  float Prowindow
//	)
//{
//
//}
int ReliableWeightLayer_gpu_forward(
		at::Tensor&  input3,  at::Tensor&  output,
		float sigma_d,    float tao_r ,  float Prowindow
		)
{

	int error = 1 ;
	if(Prowindow != 2.0f) {printf("Prowindow must be odd number 2\n"); return error;}

	//int channel = input1.size( 1);
	//if(channel!=3) return error;
	int batch = input3.size(0);
	//if(input2.size( 0) != batch) return error;

	int h = input3.size(2);
	int w = input3.size(3);
	//if(input2.size(2) != h) return error;// to add some checkpoint
	//if(input2.size(3) != w) return error;

//    int filter_size2 = input3.size( 1);
  //  int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(input3.size( 1) != 2) {printf("the flow channel should be 2\n"); return error;}
	//if(flow_weights(1) != 1){ printf("the flow_weights channel should be 1\n"); return error;}
	if(output.size(1) != 1) {printf("the flow weight channel should be 1\n"); return error;}


	//int input1_b_stride = input1.stride(0);
	//int input1_c_stride = input1.stride(1);
	//int input1_h_stride = input1.stride(2);
	//int input1_w_stride = input1.stride(3);

	///int input2_b_stride = input2.stride(0);
	//int input2_c_stride = input2.stride(1);
	//int input2_h_stride = input2.stride(2);
	//int input2_w_stride = input2.stride(3);

    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);
	
	//int flow_weights_b_stride = flow_weights.stride(0);
	//int flow_weights_c_stride = flow_weights.stride(1);
	//int flow_weights_h_stride = flow_weights.stride(2);
	//int flow_weights_w_stride = flow_weights.stride(3);	
	
	int output_b_stride = output.stride(0);
	int output_c_stride = output.stride(1);
	int output_h_stride = output.stride(2);
	int output_w_stride = output.stride(3);	
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
    //if(input1_w_stride !=1) return error;
	//if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
	//if(flow_weights_w_stride !=1) return error;
	///if(input1_b_stride != output.stride(0)) return error;
	//if(input1_c_stride != output.stride(1)) return error;

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, output);


	error = ReliableWeightLayer_gpu_forward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement,w,h, batch,  

			//input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			//flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,

			//input1,
			//input2,
			input3,
			//flow_weights,
			output,
			
			sigma_d,      tao_r ,   Prowindow
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;		 
}
int ReliableWeightLayer_gpu_backward(
		at::Tensor&  input3, 	 at::Tensor&  output,
//		at::Tensor&  output,
		at::Tensor&  gradoutput, 
		at::Tensor&  gradinput3,
        float threshhold,
		float sigma_d,    float tao_r ,  float Prowindow
	)
{

	int error = 1 ;
	if(Prowindow != 2.0f){ printf("Prowindow must be odd number 2\n"); return error;}

	//int channel = input1.size( 1);
	//if(channel!=3) return error;
	int batch = input3.size(0);
	//if(input2.size( 0) != batch) return error;
	//if(input2.size(1) != 2) return error;

	int h = input3.size(2);
	int w = input3.size(3);
	//if(input2.size(2) != h) return error;// to add some checkpoint
	//if(input2.size(3) != w) return error;


   // int filter_size2 = input3.size( 1);
    //int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(input3.size( 1) != 2) {printf("the flow channel should be 2\n"); return error;}
//	if(flow_weights(1) != 1) printf("the flow_weights channel should be 1\n"); return error;
	//if(output.size(1) != 1) printf("the flow weight channel should be 1\n"); return error;

	//int input1_b_stride = input1.stride(0);
	//int input1_c_stride = input1.stride(1);
	//int input1_h_stride = input1.stride(2);
	//int input1_w_stride = input1.stride(3);

	//int input2_b_stride = input2.stride(0);
	//int input2_c_stride = input2.stride(1);
	//int input2_h_stride = input2.stride(2);
	//int input2_w_stride = input2.stride(3);
     
    int input3_b_stride = input3.stride(0);
	int input3_c_stride = input3.stride(1);
	int input3_h_stride = input3.stride(2);
	int input3_w_stride = input3.stride(3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//int flow_weights_b_stride = flow_weights.stride(0);
	//int flow_weights_c_stride = flow_weights.stride(1);
	//int flow_weights_h_stride = flow_weights.stride(2);
	//int flow_weights_w_stride = flow_weights.stride(3);	
	
	int output_b_stride = gradoutput.stride(0);
	int output_c_stride = gradoutput.stride(1);
	int output_h_stride = gradoutput.stride(2);
	int output_w_stride = gradoutput.stride(3);		

	//TODO: do we need to assert the w_stride to be 1
	//if(input1_w_stride !=1) return error;
	//if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
    //if(input1_b_stride != gradinput1.stride(0)) return error;
	//if(input2_b_stride != gradinput2.stride(0)) return error;
	//if(input1_c_stride != gradinput1.stride(1)) return error;
	//if(input2_c_stride != gradinput2.stride(1)) return error;
	if(input3_c_stride != gradinput3.stride(1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);

	int	nElement = 0;//UNUSED  THCudaTensor_nElement(state, gradoutput);

	error  = ReliableWeightLayer_gpu_backward_kernel(
			at::globalContext().getCurrentCUDAStream(),
			nElement, //to let the nummous
			w,h, batch,  

			//input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			//flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,


			//input1,
			//input2,
			input3,
			//flow_weights,
			output,

			gradoutput,
			//gradinput1,
			//gradinput2,
			gradinput3,
			//gradflow_weights,
			threshhold,
			sigma_d,      tao_r ,   Prowindow
			);
	  if (error) {AT_ERROR("CUDA call failed");}

	return error;			
}