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

ValuePtr
OpenclNode::get_kernel (ValuePtr kvec) const
{
	Handle hkl;
	if (kvec->is_type(SECTION))
		hkl = HandleCast(kvec)->getOutgoingAtom(0);
	else
	if (kvec->is_type(SECTION_VALUE))
	{
		const ValueSeq& vsq = LinkValueCast(kvec)->value();
		hkl = HandleCast(vsq[0]);
	}

	if (nullptr == hkl or not hkl->is_type(OPENCL_KERNEL_LINK))
		throw RuntimeException(TRACE_INFO,
			"Expecting an OpenclKernelLink: got %s\n", kvec->to_string().c_str());

	// Bofus should be bof be me.
	if (this != hkl->getOutgoingAtom(0).get())
		throw RuntimeException(TRACE_INFO,
			"Cross-site scripting!: this %s\nthat: %s",
			to_string().c_str(), hkl->to_string().c_str());

	return hkl;
}

// ==============================================================

/// Find the vector length.
/// Look either for a length specification embedded in the list,
/// else obtain the shortest of all the vectors.
size_t
OpenclNode::get_vec_len(const ValueSeq& vsq, bool& have_size_spec) const
{
	have_size_spec = false;
	size_t dim = UINT_MAX;
	for (const ValuePtr& vp : vsq)
	{
		if (vp->is_type(TYPE_NODE)) continue;

		if (vp->is_type(NUMBER_NODE))
		{
			size_t sz = NumberNodeCast(vp)->size();
			if (sz < dim) dim = sz;
			continue;
		}

		if (vp->is_type(FLOAT_VALUE))
		{
			size_t sz = FloatValueCast(vp)->size();
			if (sz < dim) dim = sz;
			continue;
		}

		// Assume the length specification is wrapped like so:
		// (Connector (Number 42))
		// XXX FIXME check for insane structures here.
		if (vp->is_type(CONNECTOR))
		{
			have_size_spec = true;
			const Handle& h = HandleCast(vp)->getOutgoingAtom(0);
			double sz = NumberNodeCast(h)->get_value();
			if (sz < 0.0) continue;

			// round, just in case.
			return (size_t) (sz+0.5);
		}
	}

	return dim;
}

/// Unwrap vector.
ValuePtr
OpenclNode::get_floats(ValuePtr vp, cl::Kernel& kern,
                       size_t& pos, size_t dim) const
{
	bool from_gpu = false;
	const std::vector<double>* vals = nullptr;

	// Special-case location of the vector length specification.
	if (vp->is_type(CONNECTOR))
	{
		kern.setArg(pos, dim);
		pos ++;
		return vp;
	}

	if (vp->is_type(NUMBER_NODE))
		vals = &(NumberNodeCast(vp)->value());

	if (vp->is_type(FLOAT_VALUE))
		vals = &(FloatValueCast(vp)->value());

	// XXX For now, we ignore the type. FIXME
	if (vp->is_type(TYPE_NODE))
		from_gpu = true;

	OpenclFloatValuePtr ofv;
	if (nullptr == vals)
	{
		std::vector<double> zero;
		zero.resize(dim);
		ofv = createOpenclFloatValue(zero);
	}
	else if (vals->size() != dim)
	{
		std::vector<double> cpy(*vals);
		cpy.resize(dim);
		ofv = createOpenclFloatValue(cpy);
	}
	else
		ofv = createOpenclFloatValue(*vals);

	ofv->set_context(_device, _context);
	ofv->set_arg(kern, pos);
	if (not from_gpu)
		ofv->send_buffer();
	pos ++;
	return ofv;
}

ValueSeq
OpenclNode::make_vectors(ValuePtr kvec, cl::Kernel& kern, size_t& dim) const
{
	// Unpack kernel arguments
	ValueSeq vsq;
	if (kvec->is_type(SECTION))
	{
		const Handle& conseq = HandleCast(kvec)->getOutgoingAtom(1);
		const HandleSeq& oset = conseq->getOutgoingSet();

		// Find the shortest vector.
		for (const Handle& oh : oset)
		{
			if (oh->is_executable())
				vsq.emplace_back(oh->execute());
			else
				vsq.push_back(oh);
		}
	}
	else
	if (kvec->is_type(SECTION_VALUE))
	{
		const ValueSeq& sex = LinkValueCast(kvec)->value();
		const ValueSeq& vsx = LinkValueCast(sex[1])->value();
		for (const ValuePtr& v: vsx)
		{
			if (v->is_atom() and HandleCast(v)->is_executable())
				vsq.emplace_back(HandleCast(v)->execute());
			else
				vsq.push_back(v);
		}
	}
	else
		throw RuntimeException(TRACE_INFO,
			"Unknown data type: got %s\n", kvec->to_string().c_str());

	// Find the shortest vector.
	bool have_size_spec = false;
	dim = get_vec_len(vsq, have_size_spec);
	size_t pos = 0;
	ValueSeq flovec;
	for (const ValuePtr& v: vsq)
		flovec.emplace_back(get_floats(v, kern, pos, dim));

	// If the user never specified an explicit location in which to pass
	// the vector size, assume it is the last location. Set it now.
	// Is this a good idea? I dunno. More thinking needed.
	if (not have_size_spec)
		kern.setArg(pos, dim);

	return flovec;
}

// ==============================================================

// This job handler runs in a different thread than the main thread.
// It finishes the setup of the assorted buffers that OpenCL expects,
// sends things to the GPU, and then waits for a reply. When a reply
// is received, its turned into a FloatValue or NumberNode and handed
// to the QueueValue, where main thread can find it.
void OpenclNode::queue_job(const job_t& kjob)
{
	if (kjob._kvec->is_type(SECTION_VALUE))
	{
		// Launch kernel
		cl::Event event_handler;
		_queue.enqueueNDRangeKernel(kjob._kern,
			cl::NullRange,
			cl::NDRange(kjob._vecdim),
			cl::NullRange,
			nullptr, &event_handler);

		event_handler.wait();
		_qvp->add(kjob._kvec);
	}

#if 0
	// ------------------------------------------------------
	// Wait for results
	size_t vec_bytes = kjob._vecdim * sizeof(double);
	_queue.enqueueReadBuffer(kjob._outvec->get_buffer(),
		CL_TRUE, 0, vec_bytes, kjob._outvec->data(),
		nullptr, &event_handler);
	event_handler.wait();

	// XXX TODO: we should probably wrap this with the kvec, so that
	// the user knows who these results belong to. I guess using an
	// ArrowLink, right?
	_qvp->add(kjob._outvec);
#endif
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

	ValuePtr hkl = get_kernel(kvec);
	OpenclKernelLinkPtr okp = OpenclKernelLinkCast(hkl);
	cl::Kernel kern = okp->get_kernel();

	size_t dim = 0;
	ValueSeq flovecs = make_vectors (kvec, kern, dim);
	ValuePtr args = createLinkValue(flovecs);
	ValuePtr jobvec = createLinkValue(SECTION_VALUE, ValueSeq{hkl, args});

	job_t kjob;
	kjob._kvec = jobvec;
	kjob._kern = kern;
	kjob._vecdim = dim;

	// Send everything off to the GPU.
	_dispatch_queue.enqueue(kjob);
}

// ==============================================================

// Adds factory when library is loaded.
DEFINE_NODE_FACTORY(OpenclNode, OPENCL_NODE);

// ====================================================================

void opencog_opencl_init(void)
{
   // Force shared lib ctors to run
};
