
#ifdef __cplusplus
extern "C" {
#endif

int SeparableConvFlowLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
	//	const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		const int flow_output_b_stride, const int flow_output_c_stride, const int flow_output_h_stride, const int flow_output_w_stride,

		at::Tensor& input1,    		at::Tensor& input2,    	at::Tensor& input3, 	 at::Tensor& flow_output

		);

int SeparableConvFlowLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch, const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
	//	const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		const int flow_output_b_stride, const int flow_output_c_stride, const int flow_output_h_stride, const int flow_output_w_stride,

		at::Tensor& input1,        		at::Tensor& input2,		at::Tensor& input3,

		at::Tensor& gradflow_output,    		at::Tensor& gradinput1,  		at::Tensor& gradinput2,  		at::Tensor& gradinput3
		);
		
int SeparableConvLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor& input1,    		at::Tensor& input2,    	at::Tensor& input3, 	at::Tensor& output

		);

int SeparableConvLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch, const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor& input1,        		at::Tensor& input2,		at::Tensor& input3,

		at::Tensor& gradoutput,    		at::Tensor& gradinput1,  		at::Tensor& gradinput2,  		at::Tensor& gradinput3
		);


int InterpolationLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int batch,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,

		at::Tensor& input1,
		at::Tensor& input2,
		at::Tensor& output

		);

int InterpolationLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int batch,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,

		at::Tensor& input1,
		at::Tensor& input2,
		at::Tensor& gradoutput,
		at::Tensor& gradinput1,
		at::Tensor& gradinput2
		);


int InterpolationChLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int batch,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,

		at::Tensor& input1,
		at::Tensor& input2,
		at::Tensor& output

		);

int InterpolationChLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int batch,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,

		at::Tensor& input1,
		at::Tensor& input2,
		at::Tensor& gradoutput,
		at::Tensor& gradinput1,
		at::Tensor& gradinput2
		);
int FilterInterpolationLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const  int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,

		at::Tensor& input1,    		at::Tensor& input2,    	at::Tensor& input3, 	at::Tensor& output

		);

int FilterInterpolationLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch,    		const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,

		at::Tensor& input1,        		at::Tensor& input2,		at::Tensor& input3,

		at::Tensor& gradoutput,    		at::Tensor& gradinput1,  		at::Tensor& gradinput2,  		at::Tensor& gradinput3
		);


int FlowProjection_gpu_forward_kernel(
		cudaStream_t stream, 		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const int fillhole,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		at::Tensor& input1,
		at::Tensor& count,
		at::Tensor& output

		);

int FlowProjection_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int batch,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		at::Tensor& input1,
		at::Tensor& count,
		at::Tensor& gradoutput,
		at::Tensor& gradinput1
		);


int WeightedFlowProjection_gpu_forward_kernel(
		cudaStream_t stream, 		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const int fillhole, const float threshhold,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
        const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
        const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,
		const int weight_b_stride,const int weight_c_stride, const int weight_h_stride, const int weight_w_stride,

		at::Tensor& input1,at::Tensor& input2, at::Tensor& input3,
		at::Tensor& count, at::Tensor& weight,
		at::Tensor& output

		);

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

		at::Tensor& input1,at::Tensor& input2, at::Tensor& input3,
		at::Tensor& count,const at::Tensor&  weight,
		at::Tensor& gradoutput,
		at::Tensor& gradinput1
		);

int WeightLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch,  

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
//		const int flow1_grad_b_stride,const int flow1_grad_c_stride,const int flow1_grad_h_stride,const int flow1_grad_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor& input1,    		at::Tensor& input2,    	at::Tensor& input3,
//		const at::Tensor&  flow1_grad,
        at::Tensor& output,
		float  lambda_e, float	lambda_v, float Nw
		);

int WeightLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch,    		 

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
//		const int flow1_grad_b_stride,const int flow1_grad_c_stride,const int flow1_grad_h_stride,const int flow1_grad_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor& input1,     	at::Tensor& input2, at::Tensor& input3,
//		const at::Tensor&  flow1_grad,
        at::Tensor& output,

		at::Tensor& gradoutput, at::Tensor& gradinput1, at::Tensor& gradinput2, at::Tensor& gradinput3,
//		 at::Tensor& gradflow1_grad,

		float  lambda_e, float	lambda_v, float Nw

		);


int PixelValueLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch,  

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor& input1, at::Tensor& input3, const at::Tensor&  flow_weights,	at::Tensor& output,
		float	sigma_d,     float tao_r , float  Prowindow
		);

int PixelValueLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch,    		 

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor& input1,  at::Tensor& input3, const at::Tensor&  flow_weights, 

		at::Tensor& gradoutput, at::Tensor&  gradinput1,   at::Tensor&  gradinput3, at::Tensor& gradflow_weights,
		float	sigma_d,     float tao_r , float  Prowindow

		);		


int PixelWeightLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		 	const int batch,  

		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor& input3, const at::Tensor&  flow_weights,	at::Tensor& output,
		float	sigma_d,     float tao_r , float  Prowindow
		);

int PixelWeightLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    	 		const int batch,    		 

		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		
		at::Tensor& input3, const at::Tensor&  flow_weights,const	 at::Tensor& output,
		at::Tensor& gradoutput,  at::Tensor& gradinput3, at::Tensor& gradflow_weights,
        float threshhold,
		float	sigma_d,     float tao_r , float  Prowindow

		);

int ReliableWeightLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 	 		const int batch,  

		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		at::Tensor& input3,  	at::Tensor& output,
		float	sigma_d,     float tao_r , float  Prowindow
		);

int ReliableWeightLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    	 		const int batch,    		 

		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		
		at::Tensor& input3,   
        at::Tensor& output,
		at::Tensor& gradoutput,  at::Tensor& gradinput3,
        float threshhold,
		float	sigma_d,     float tao_r , float  Prowindow

		);
		
#ifdef __cplusplus
}
#endif

