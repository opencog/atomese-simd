/**
 * OpenCL float-point math demo.
 *
 * Demo of accumulating into fixed location.
 *
 * Copyright (c) 2025 Linas Vepstas
 */

#include "scaffolding.h"

// Declare two floating point vectors. One acts as an accumulator;
// the other is constantly changing.
void run_accum(cl::Device ocldev, cl::Context context, cl::Program program)
{
	size_t vec_dim = 6;
	std::vector<double> accum(vec_dim);
	std::vector<double> b(vec_dim);

	size_t vec_bytes = vec_dim * sizeof(double);

	// Buffers holding data that will go to the GPU's.
	// Data is copied in; changes after this point have no effect.
	// The SVM (Shared Virtual Memory) extension avoids copyin,
	// but SVM requires OpenCL 2.0 for support.
	cl::Buffer vecaccum(context,
		CL_MEM_READ_WRITE, vec_bytes);

	cl::Buffer vecb(context,
		CL_MEM_READ_ONLY, vec_bytes);

	// Initialize accumulator to zero.
	for (size_t i=0; i<vec_dim; i++)
		accum[i] = 0.0;

	cl::CommandQueue queue(context, ocldev);
	cl::Event event_handler;
	queue.enqueueWriteBuffer(vecaccum, CL_TRUE, 0,
		vec_bytes, accum.data(), nullptr, &event_handler);

	event_handler.wait();
	fprintf(stderr, "Done initializing the accumulator.\n");

	// Set up the stream.
	for (size_t i=0; i<vec_dim; i++)
		b[i] = 1.0;

	queue.enqueueWriteBuffer(vecb, CL_TRUE, 0,
		vec_bytes, b.data(), nullptr, &event_handler);

	// The program to run on the GPU, and the arguments it takes.
	cl::Kernel kernel(program, "vec_add");
	kernel.setArg(0, vecaccum);
	kernel.setArg(1, vecaccum);
	kernel.setArg(2, vecb);
	kernel.setArg(3, vec_dim);

	// Launch
	queue.enqueueNDRangeKernel(kernel,
		cl::NullRange,
		cl::NDRange(vec_dim),
		cl::NullRange,
		nullptr, &event_handler);

	event_handler.wait();
	fprintf(stderr, "Done waiting on exec\n");

	queue.enqueueReadBuffer(vecaccum, CL_TRUE, 0,
		vec_bytes, accum.data(), nullptr, &event_handler);
	event_handler.wait();
	fprintf(stderr, "Done reading accumulator\n");

	printf("The accumulator is:\n");
	for (size_t i=0; i<vec_dim; i++)
		printf("%ld = %f\n", i, accum[i]);

	// -------------------------------------------------------------
	// Do it again
	// Set up the stream.
	for (size_t i=0; i<vec_dim/2; i++)
		b[i] = 2.0;

	queue.enqueueWriteBuffer(vecb, CL_TRUE, 0,
		vec_bytes, b.data(), nullptr, &event_handler);

	// Launch
	queue.enqueueNDRangeKernel(kernel,
		cl::NullRange,
		cl::NDRange(vec_dim),
		cl::NullRange,
		nullptr, &event_handler);

	event_handler.wait();
	queue.enqueueReadBuffer(vecaccum, CL_TRUE, 0,
		vec_bytes, accum.data(), nullptr, &event_handler);
	event_handler.wait();
	printf("The accumulator is:\n");
	for (size_t i=0; i<vec_dim; i++)
		printf("%ld = %f\n", i, accum[i]);


}

// Run code on the GPU's.
int main(int argc, char* argv[])
{
	// Pick the first device found.
	// cl::Device ocldev = find_device("", "AMD");
	cl::Device ocldev = find_device("", "");
	std::string dname = ocldev.getInfo<CL_DEVICE_NAME>();
	printf("Will use: %s\n", dname.c_str());

	cl::Context ctxt(ocldev);
	cl::Program prog;
	build_kernel(ctxt, "vec-mult.cl", prog);
	run_accum(ocldev, ctxt, prog);
}
