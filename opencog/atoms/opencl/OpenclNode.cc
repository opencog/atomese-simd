/*
 * opencog/atoms/opencl/OpenclNode.cc
 *
 * Copyright (C) 2025 Linas Vepstas
 * All Rights Reserved
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program; if not, write to:
 * Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <iostream>
#include <fstream>

#include <opencog/util/exceptions.h>
#include <opencog/util/Logger.h>
#include <opencog/util/oc_assert.h>
#include <opencog/atomspace/AtomSpace.h>
#include <opencog/atoms/base/Link.h>
#include <opencog/atoms/base/Node.h>
#include <opencog/atoms/core/NumberNode.h>
#include <opencog/atoms/value/FloatValue.h>
#include <opencog/atoms/value/LinkValue.h>
#include <opencog/atoms/value/StringValue.h>
#include <opencog/atoms/value/VoidValue.h>
#include <opencog/atoms/value/ValueFactory.h>

#include <opencog/opencl/types/atom_types.h>
#include <opencog/sensory/types/atom_types.h>
#include "OpenclFloatValue.h"
#include "OpenclNode.h"

using namespace opencog;

OpenclNode::OpenclNode(const std::string&& str) :
	StreamNode(OPENCL_NODE, std::move(str)),
	_dispatch_queue(this, &OpenclNode::queue_job, 1)
{
	init();
}

OpenclNode::OpenclNode(Type t, const std::string&& str) :
	StreamNode(t, std::move(str)),
	_dispatch_queue(this, &OpenclNode::queue_job, 1)
{
	if (not nameserver().isA(t, OPENCL_NODE))
		throw RuntimeException(TRACE_INFO,
			"Expecting OpenclNode, got %s\n", to_string().c_str());

	init();
}

OpenclNode::~OpenclNode()
{
}

#define BAD_URL \
	throw RuntimeException(TRACE_INFO, \
		"Unsupported URL \"%s\"\n" \
		"\tExpecting 'opencl://platform:device/file/path/kernel.cl'", \
		url.c_str());

/// Validate the OpenCL URL
void OpenclNode::init(void)
{
	const std::string& url = get_name();
	if (0 != url.compare(0, 9, "opencl://")) BAD_URL;

	// Ignore the first 9 chars "opencl://"
	size_t pos = 9;

	// Extract platform name substring
	size_t platend = url.find(':', pos);
	if (std::string::npos == platend) BAD_URL;
	if (pos < platend)
	{
		_splat = url.substr(pos, platend-pos);
		pos = platend;
	}
	pos ++;

	// Extract device name substring
	size_t devend = url.find('/', pos);
	if (std::string::npos == devend) BAD_URL;
	if (pos < devend)
	{
		_sdev = url.substr(pos, devend-pos);
		pos = devend;
	}
	_filepath = url.substr(pos);

	// What kind of file is it? Source or SPV?
	pos = url.find_last_of('.');
	if (std::string::npos == pos) BAD_URL;

	_is_spv = (url.substr(pos) == ".spv");
}

// ==============================================================

void OpenclNode::find_device(void)
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	for (const auto& plat : platforms)
	{
		std::string pname = plat.getInfo<CL_PLATFORM_NAME>();
		if (0 < _splat.size() and pname.find(_splat) == std::string::npos)
			continue;

		std::vector<cl::Device> devices;
		plat.getDevices(CL_DEVICE_TYPE_ALL, &devices);
		for (const cl::Device& dev: devices)
		{
			std::string dname = dev.getInfo<CL_DEVICE_NAME>();
			if (dname.find(_sdev) == std::string::npos)
				continue;

			_platform = plat;
			_device = dev;

			logger().info("OpenclNode: Using platform '%s' and device '%s'\n",
				pname.c_str(), dname.c_str());

			return;
		}
	}

	throw RuntimeException(TRACE_INFO,
		"Unable to find platform:device in URL \"%s\"\n",
		get_name().c_str());
}

// ==============================================================

void OpenclNode::build_kernel(void)
{
	// Copy in source code. Must be a better way!?
	std::ifstream srcfm(_filepath);
	std::string src(std::istreambuf_iterator<char>(srcfm),
		(std::istreambuf_iterator<char>()));

	if (0 == src.size())
		throw RuntimeException(TRACE_INFO,
			"Unable to find source file in URL \"%s\"\n",
			get_name().c_str());

	cl::Program::Sources sources;
	sources.push_back(src);

	_program = cl::Program(_context, sources);

	// Compile
	try
	{
		// Specifying flags causes exception.
		// program.build("-cl-std=CL1.2");
		_program.build("");
	}
	catch (const cl::Error& e)
	{
		logger().info("OpenclNode failed compile >>%s<<\n",
			_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(_device).c_str());
		throw RuntimeException(TRACE_INFO,
			"Unable to compile source file in URL \"%s\"\n",
				get_name().c_str());
	}
}

// ==============================================================

void OpenclNode::load_kernel(void)
{
	// Copy in SPV file. Must be a better way!?
	std::ifstream spvfm(_filepath);
	std::string spv(std::istreambuf_iterator<char>(spvfm),
		(std::istreambuf_iterator<char>()));

	if (0 == spv.size())
		throw RuntimeException(TRACE_INFO,
			"Unable to find SPV file in URL \"%s\"\n",
			get_name().c_str());

	_program = cl::Program(_context, spv);
}

// ==============================================================

/// Attempt to open connection to OpenCL device
void OpenclNode::open(const ValuePtr& out_type)
{
	if (_qvp)
		throw RuntimeException(TRACE_INFO,
			"Device already open! %s\n", get_name().c_str());

	StreamNode::open(out_type);
	if (not nameserver().isA(_item_type, FLOAT_VALUE) and
		 not nameserver().isA(_item_type, NUMBER_NODE))
		throw RuntimeException(TRACE_INFO,
			"Expecting the type to be a FloatValue or NumberNode; got %s\n",
			out_type->to_string().c_str());

	// Try to create the OpenCL device
	find_device();
	_context = cl::Context(_device);
	_queue = cl::CommandQueue(_context, _device);

	// Try to load source or spv file
	if (_is_spv)
		load_kernel();
	else
		build_kernel();

	_qvp = createQueueValue();
}

bool OpenclNode::connected(void) const
{
	return nullptr != _qvp;
}

void OpenclNode::close(const ValuePtr& ignore)
{
	if (_qvp)
		_qvp->close();
	_qvp = nullptr;

	// XXX more to do here. FIXME
}

// ==============================================================

#if LATER_SOMEDAY
Handle _global_desc = Handle::UNDEFINED;

void OpenclNode::do_describe(void)
{
	if (_global_desc) return;

	HandleSeq cmds;

	// Describe exactly how to Open this stream.
	// It needs no special arguments.
	Handle open_cmd =
		make_description("Open connection to GPU",
		                 "OpenLink", "OpenclNode");
	cmds.emplace_back(open_cmd);

	// Write  XXX this is wrong.
	Handle write_cmd =
		make_description("Write kernel and data to GPU",
		                 "WriteLink", "ItemNode");
	cmds.emplace_back(write_cmd);

	_global_desc = createLink(cmds, CHOICE_LINK);
}

// This is totally bogus because it is unused.
// This should be class static member
ValuePtr OpenclNode::describe(AtomSpace* as, bool silent)
{
	if (_description) return as->add_atom(_description);
	_description = as->add_atom(_global_desc);
	return _description;
}
#endif

// ==============================================================

ValuePtr OpenclNode::stream(void) const
{
	if (not connected())
		throw RuntimeException(TRACE_INFO,
			"Device not open! %s\n", get_name().c_str());

	return _qvp;
}

ValuePtr OpenclNode::read(void) const
{
	if (not connected())
		throw RuntimeException(TRACE_INFO,
			"Device not open! %s\n", get_name().c_str());

	return _qvp->remove();
}

// This job handler runs in a different thread than the main thread.
// It finishes the setup of the assorted buffers that OpenCL expects,
// sends things to the GPU, and then waits for a reply. When a reply
// is received, its turned into a FloatValue or NumberNode and handed
// to the QueueValue, where main thread can find it.
void OpenclNode::queue_job(const job_t& kjob)
{
	// Copy vectors into cl::Buffer
	size_t vec_bytes = kjob._vec_dim * sizeof(double);
	for (const double* flt : kjob._flts)
	{
		kjob._invec.emplace_back(
			cl::Buffer(_context,
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				vec_bytes,
				(void*) flt));
	}

	// XXX Hardwired assumption about argument order.
	// FIXME... but how ???
	// The problem is this assumes the output comes first
	// in the kernel, followed by the arguments. It could be
	// different.
	OpenclFloatValuePtr ofv = createOpenclFloatValue(vec_bytes);
	ofv->set_context(_context);
	ofv->set_arg(kjob._kernel, 0, true);
	for (size_t i=1; i<kjob._ninputs; i++)
		kjob._kernel.setArg(i, kjob._invec[i-1]);

	// XXX This is the wrong thing to do in the long run.
	// Or is it? each kernel gets its own size ... what's the problem?
	kjob._kernel.setArg(kjob._ninputs, kjob._vec_dim);

	// Launch kernel
	cl::Event event_handler;
	_queue.enqueueNDRangeKernel(kjob._kernel,
		cl::NullRange,
		cl::NDRange(kjob._vec_dim),
		cl::NullRange,
		nullptr, &event_handler);

	event_handler.wait();

	// ------------------------------------------------------
	// Wait for results
	_queue.enqueueReadBuffer(ofv->get_buffer(),
		CL_TRUE, 0, vec_bytes, ofv->data(),
		nullptr, &event_handler);
	event_handler.wait();

	// XXX TODO: we should probably wrap this with the kvec, so that
	// the user knows who these results belong to. I guess using an
	// ExecutionLink, right?
	_qvp->add(std::move(ofv));
}

// ==============================================================

/// Unwrap kernel name.
const std::string&
OpenclNode::get_kern_name (ValuePtr vp) const
{
	if (vp->is_atom() and HandleCast(vp)->is_executable())
		vp = HandleCast(vp)->execute();

	if (vp->is_node())
		return HandleCast(vp)->get_name();

	if (vp->is_type(STRING_VALUE))
		return StringValueCast(vp)->value()[0];

	throw RuntimeException(TRACE_INFO,
		"Expecting Value with kernel name, got %s\n",
		vp->to_string().c_str());
}

/// Unwrap vector.
const std::vector<double>&
OpenclNode::get_floats (ValuePtr vp, size_t& dim) const
{
	if (vp->is_atom() and HandleCast(vp)->is_executable())
		vp = HandleCast(vp)->execute();

	if (vp->is_type(NUMBER_NODE))
	{
		const std::vector<double>& vals(NumberNodeCast(HandleCast(vp))->value());
		if (vals.size() < dim) dim = vals.size();
		return vals;
	}

	if (vp->is_type(FLOAT_VALUE))
	{
		const std::vector<double>& vals(FloatValueCast(vp) ->value());
		if (vals.size() < dim) dim = vals.size();
		return vals;
	}

	throw RuntimeException(TRACE_INFO,
		"Expecting FloatValue or NumberNode, got %s\n",
		vp->to_string().c_str());
}

// ==============================================================
// Send kernel and data

void OpenclNode::write_one(const ValuePtr& kvec)
{
	do_write(kvec);
}

// Prep everything needed to be able to send off a job to the GPU.
// The code here does everything that might result in an exception
// being thrown, i.e. due to user errors (e.g. badly written Atomese)
// The actual communications with the GPU is done in a distinct thread,
// so that the main thread does not hang, waiting for results to arrive.
void OpenclNode::do_write(const ValuePtr& kvec)
{
	if (0 == kvec->size())
		throw RuntimeException(TRACE_INFO,
			"Expecting a kernel name, got %s\n", kvec->to_string().c_str());

	job_t kjob;
	kjob._kvec = kvec;

	// Unpack kernel name and kernel arguments
	std::string kern_name;
	kjob._vec_dim = UINT_MAX;
	if (kvec->is_type(LIST_LINK))
	{
		const HandleSeq& oset = HandleCast(kvec)->getOutgoingSet();
		kern_name = get_kern_name(oset[0]);

		// Find the shortest vector.
		for (size_t i=1; i<oset.size(); i++)
			kjob._flts.emplace_back(get_floats(oset[i], kjob._vec_dim).data());
	}
	else
	if (kvec->is_type(LINK_VALUE))
	{
		const ValueSeq& vsq = LinkValueCast(kvec)->value();
		kern_name = get_kern_name(vsq[0]);

		// Find the shortest vector.
		for (size_t i=1; i<vsq.size(); i++)
			kjob._flts.emplace_back(get_floats(vsq[i], kjob._vec_dim).data());
	}
	else
		throw RuntimeException(TRACE_INFO,
			"Unknown data type: got %s\n", kvec->to_string().c_str());

	// XXX TODO this will throw exception if user mis-typed the
	// kernel name. We should catch this and print a friendlier
	// error message.
	kjob._kernel = cl::Kernel(_program, kern_name.c_str());

	kjob._ninputs = kvec->size();

	// Send everything off to the GPU.
	_dispatch_queue.enqueue(std::move(kjob));
}

// ==============================================================

// Adds factory when library is loaded.
DEFINE_NODE_FACTORY(OpenclNode, OPENCL_NODE);

// ====================================================================

void opencog_opencl_init(void)
{
   // Force shared lib ctors to run
};
