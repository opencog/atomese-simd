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
#include <opencog/atoms/base/Node.h>
#include <opencog/atoms/value/ValueFactory.h>

#include <opencog/opencl/types/atom_types.h>
#include <opencog/sensory/types/atom_types.h>
#include "OpenclFloatValue.h"
#include "OpenclJobValue.h"
#include "OpenclKernelLink.h"
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
	// One of the TODO's is to crawl over the icoming set, look for
	// OpenclKernelLinks and tell them to shut down too.
}

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

// ==============================================================

// This job handler runs in a different thread than the main thread.
// It finishes the setup of the assorted buffers that OpenCL expects,
// sends things to the GPU, and then waits for a reply. When a reply
// is received, its turned into a FloatValue or NumberNode and handed
// to the QueueValue, where main thread can find it.
void OpenclNode::queue_job(const ValuePtr& vp)
{
	// XXX TODO All of these should probably share the same
	// cl::Event (even though they can run on different queues.)
	// Or maybe one Queue would be better ...
	if (vp->is_type(OPENCL_JOB_VALUE))
	{
		OpenclJobValuePtr ojv = OpenclJobValueCast(vp);
		ojv->run();
		_qvp->add(ojv);
		return;
	}

	// If told to write a vector, then we upload that vector data
	// to the GPU.
	if (vp->is_type(OPENCL_DATA_VALUE))
	{
		OpenclFloatValuePtr ofv = OpenclFloatValueCast(vp);
		ofv->set_context(_device, _context);
		ofv->send_buffer();
		_qvp->add(ofv);
		return;
	}
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
void OpenclNode::do_write(const ValuePtr& vp)
{
	// Ready-to-go. Dispatch.
	if (vp->is_type(OPENCL_DATA_VALUE) or
	    vp->is_type(OPENCL_JOB_VALUE))
	{
		_dispatch_queue.enqueue(vp);
		return;
	}

	if (vp->is_type(SECTION))
	{
		OpenclJobValuePtr kern = createOpenclJobValue(HandleCast(vp));
		kern->build(get_handle());
		_dispatch_queue.enqueue(kern);
		return;
	}

	throw RuntimeException(TRACE_INFO,
		"Expecting data or a job, got %s\n", vp->to_string().c_str());
}

// ==============================================================

// Adds factory when library is loaded.
DEFINE_NODE_FACTORY(OpenclNode, OPENCL_NODE);

// ====================================================================

void opencog_opencl_init(void)
{
   // Force shared lib ctors to run
};
