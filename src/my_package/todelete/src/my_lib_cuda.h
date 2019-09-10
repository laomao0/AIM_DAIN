
int SeparableConvFlowLayer_gpu_forward(
		at::Tensor& input1,
		at::Tensor& input2,
		at::Tensor& input3,
		//at::Tensor& output,
		at::Tensor& flow_output
		);

int SeparableConvFlowLayer_gpu_backward(
		at::Tensor& input1,
		at::Tensor& input2,
		at::Tensor& input3,
		at::Tensor& gradflow_output,
		at::Tensor& gradinput1,
		at::Tensor& gradinput2,
		at::Tensor& gradinput3
		); 
int SeparableConvLayer_gpu_forward(
		at::Tensor& input1,
		at::Tensor& input2,
		at::Tensor& input3,
		at::Tensor& output
		);

int SeparableConvLayer_gpu_backward(
		at::Tensor& input1,
		at::Tensor& input2,
		at::Tensor& input3,
		at::Tensor& gradoutput,
		at::Tensor& gradinput1,
		at::Tensor& gradinput2,
		at::Tensor& gradinput3
		);


int InterpolationLayer_gpu_forward(
		at::Tensor& input1,
		at::Tensor& input2,
		at::Tensor& output

		);

int InterpolationLayer_gpu_backward(
		THCudaTensor* input1,
		THCudaTensor* input2,

		at::Tensor& gradoutput,

		at::Tensor& gradinput1,
		at::Tensor& gradinput2
		);
int InterpolationChLayer_gpu_forward(
		at::Tensor& input1,
		at::Tensor& input2,
		at::Tensor& output

		);

int InterpolationChLayer_gpu_backward(
		THCudaTensor* input1,
		THCudaTensor* input2,

		at::Tensor& gradoutput,

		at::Tensor& gradinput1,
		at::Tensor& gradinput2
		);
int FilterInterpolationLayer_gpu_forward(
		at::Tensor& input1,
		at::Tensor& input2,
		at::Tensor& input3,
		at::Tensor& output

		);

int FilterInterpolationLayer_gpu_backward(
		at::Tensor& input1,
		at::Tensor& input2,
		at::Tensor& input3,
		at::Tensor& gradoutput,
		at::Tensor& gradinput1,
		at::Tensor& gradinput2,
		at::Tensor& gradinput3
		);

int FlowProjectionLayer_gpu_forward(
		at::Tensor& input1,
		at::Tensor& count,
		at::Tensor& output,
		int fillhole
		);

int FlowProjectionLayer_gpu_backward(
		at::Tensor& input1,
        at::Tensor& count,
		at::Tensor& gradoutput,
		at::Tensor& gradinput1
		);

int DepthFlowProjectionLayer_gpu_forward(
		at::Tensor& input1,
		at::Tensor& input2,
		at::Tensor& count,
		at::Tensor& output,
		int fillhole
		);

int DepthFlowProjectionLayer_gpu_backward(
		at::Tensor& input1,
		at::Tensor& input2,
        at::Tensor& count,
		at::Tensor& output,
        at::Tensor& gradoutput,
		at::Tensor& gradinput1,
		at::Tensor& gradinput2
		);

int WeightedFlowProjectionLayer_gpu_forward(
		at::Tensor& input1,
		at::Tensor& input2,
		THCudaTensor  * input3,
		at::Tensor& count,
		at::Tensor& weight,
		at::Tensor& output,
		int fillhole,
		float threshhold
		);

int WeightedFlowProjectionLayer_gpu_backward(
		at::Tensor& input1,
		at::Tensor& input2,
		at::Tensor& input3,
        at::Tensor& count,
        at::Tensor& weight,
		at::Tensor& gradoutput,
		at::Tensor& gradinput1,
		float threshhold
		);
//int FlowFillholelayer_gpu_forward(
//		at::Tensor& input1,
//		at::Tensor& output
//		);



int WeightLayer_gpu_forward(
		at::Tensor& input1, 		at::Tensor& input2, 		at::Tensor& input3,
//		at::Tensor& flow1_grad,
	    at::Tensor& output,
		float  lambda_e, float	lambda_v, float Nw
		);
int WeightLayer_gpu_backward(
		at::Tensor& input1, 		at::Tensor& input2, 		at::Tensor& input3,
//		 at::Tensor& flow1_grad,
        at::Tensor& output,
		at::Tensor& gradoutput, 
		at::Tensor& gradinput1,		at::Tensor& gradinput2, 	at::Tensor& gradinput3,
//		at::Tensor&  gradflow1_grad,
		float  lambda_e, float	lambda_v, float Nw
		);
		
int PixelValueLayer_gpu_forward(
		at::Tensor& input1,  at::Tensor& input3, 	at::Tensor& flow_weights, at::Tensor& output,
		float sigma_d,    float tao_r ,  float Prowindow
		);
int PixelValueLayer_gpu_backward(
		at::Tensor& input1,  		at::Tensor& input3, 	at::Tensor& flow_weights, 		 
		at::Tensor& gradoutput, 
		at::Tensor& gradinput1,		at::Tensor& gradinput3, at::Tensor& gradflow_weights,
		float sigma_d,    float tao_r ,  float Prowindow
		);
int PixelWeightLayer_gpu_forward(
		at::Tensor& input3, 	at::Tensor& flow_weights, 		at::Tensor& output,
		float sigma_d,    float tao_r ,  float Prowindow
		);
int PixelWeightLayer_gpu_backward(
		at::Tensor& input3, 	at::Tensor& flow_weights,  at::Tensor& output,
		at::Tensor& gradoutput, 
		at::Tensor& gradinput3, at::Tensor& gradflow_weights,
        float threshhold,
		float sigma_d,    float tao_r ,  float Prowindow
		);
		
//int ReliableValueLayer_gpu_forward(
//		at::Tensor& input3, 	at::Tensor& flow_weights, 		at::Tensor& output,
//		float sigma_d,    float tao_r ,  float Prowindow
//		); 
//int ReliableValueLayer_gpu_backward(
//		at::Tensor& input3, 	at::Tensor& flow_weights, 		at::Tensor& output,
//		at::Tensor& gradinput3, at::Tensor& gradflow_weights,
//		float sigma_d,    float tao_r ,  float Prowindow
//	);
int ReliableWeightLayer_gpu_forward(
		at::Tensor& input3,  at::Tensor& output,
		float sigma_d,    float tao_r ,  float Prowindow
		); 
int ReliableWeightLayer_gpu_backward(
		at::Tensor& input3, 	 	 at::Tensor& output,
		at::Tensor& gradoutput, 
		at::Tensor& gradinput3,  
				float threshhold,
		float sigma_d,    float tao_r ,  float Prowindow
	); 