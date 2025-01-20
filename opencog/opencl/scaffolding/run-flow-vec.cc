/**
 * OpenCL data flow demo.
 *
 * Simple demo of streaming float point data to/from GPU hardware.
 *
 * Copyright (c) 2025 Linas Vepstas
 */

#include "scaffolding.h"

// Wrapper: remote service, provided by GPU located at the far end of a
// channel. Accepts two vectors as input, returns one as output.
class VectorService
{
protected:
	cl::Context _context;
	cl::CommandQueue _queue;
	cl::Kernel _kernel;

	size_t _vec_dim;
	cl::Buffer _invec_a;
	cl::Buffer _invec_b;
	cl::Buffer _outvec;

public:
	VectorService(cl::Context context, cl::CommandQueue q)
	{
		_context = context;
		_queue = q;
	}

	void setup(cl::Program program, const char* name, size_t vec_dim)
	{
		_vec_dim = vec_dim;

		size_t vec_bytes = _vec_dim * sizeof(double);

		_invec_a = cl::Buffer(_context, CL_MEM_READ_ONLY, vec_bytes);
		_invec_b = cl::Buffer(_context, CL_MEM_READ_ONLY, vec_bytes);
		_outvec = cl::Buffer(_context, CL_MEM_READ_WRITE, vec_bytes);

		// The program to run on the GPU, and the arguments it takes.
		_kernel = cl::Kernel(program, name);
		_kernel.setArg(0, _outvec);
		_kernel.setArg(1, _invec_a);
		_kernel.setArg(2, _invec_b);
		_kernel.setArg(3, _vec_dim);
	}

	void launch(std::vector<double>& a, std::vector<double>& b)
	{
		size_t vec_bytes = _vec_dim * sizeof(double);
		cl::Event event_handler;

		// Copyin data
		_queue.enqueueWriteBuffer(_invec_a, CL_TRUE, 0,
			vec_bytes, a.data(), nullptr, &event_handler);

		_queue.enqueueWriteBuffer(_invec_b, CL_TRUE, 0,
			vec_bytes, b.data(), nullptr, &event_handler);

		// Actually run the code
		_queue.enqueueNDRangeKernel(_kernel,
			cl::NullRange,
			cl::NDRange(_vec_dim),
			cl::NullRange,
			nullptr, &event_handler);

		event_handler.wait();
		fprintf(stderr, "Done waiting on exec\n");
	}

	void get_results(std::vector<double>& results)
	{
		size_t vec_bytes = _vec_dim * sizeof(double);
		cl::Event event_handler;

		_queue.enqueueReadBuffer(_outvec, CL_TRUE, 0,
			vec_bytes, results.data(), nullptr, &event_handler);
		event_handler.wait();

		printf("The results:\n");
		for (size_t i=0; i<_vec_dim; i++)
			printf("result[%ld] = %f\n", i, results[i]);
	}
};

void run_flow (cl::Device ocldev,
               cl::Context context,
               cl::Program program)
{
	cl::CommandQueue queue(context, ocldev);

	VectorService vsrv(context, queue);

	// size_t vec_dim = 64;
	size_t vec_dim = 10;
	vsrv.setup(program, "vec_mult", vec_dim);

	// Set up vectors
	std::vector<double> a(vec_dim);
	std::vector<double> b(vec_dim);
	std::vector<double> prod(vec_dim);

	// Product will be triangle numbers.
	for (size_t i=0; i<vec_dim; i++)
	{
		a[i] = (double) i;
		b[i] = 0.5 * (double) i+1;
		prod[i] = 0.0;
	}

	vsrv.launch(a, b);
	vsrv.get_results(prod);

	// Product will be even numbers.
	for (size_t i=0; i<vec_dim; i++)
	{
		a[i] = 2.0;
		b[i] = (double) i;
		prod[i] = 0.0;
	}

	vsrv.launch(a, b);
	vsrv.get_results(prod);
}

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
	// load_kernel(ctxt, "vec-mult.spv", prog);
	run_flow(ocldev, ctxt, prog);
}
