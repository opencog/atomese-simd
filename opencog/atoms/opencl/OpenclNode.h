/*
 * opencog/atoms/opencl/OpenclNode.h
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

#ifndef _OPENCOG_OPENCL_NODE_H
#define _OPENCOG_OPENCL_NODE_H

#include <opencog/util/async_method_caller.h>
#include <opencog/atoms/value/QueueValue.h>
#include <opencog/atoms/sensory/StreamNode.h>
#include <opencog/atoms/opencl/OpenclFloatValue.h>
#include <opencog/atoms/opencl/opencl-headers.h>

namespace opencog
{

/** \addtogroup grp_atomspace
 *  @{
 */

/**
 * OpenclNodes hold an ordered vector of doubles.
 */
class OpenclNode
	: public StreamNode
{
protected:
	void init(void);

	// URL specifying platform and device.
	std::string _splat; // platform substring
	std::string _sdev;  // device substring
	std::string _filepath; // path to cl, clcpp or spv file
	bool _is_spv; // true if a *.spv file

	// Actual platform and device to connect to.
	void find_device(void);
	cl::Platform _platform;
	cl::Device _device;

	// Execution context.
	// (TODO: I guess we could have several of these per device!?)
	cl::Context _context;

	// Async I/O queue to the execution context.
	// (TODO: I guess we could have several of these per context!?)
	cl::CommandQueue _queue;

	// Kernel compilation
	void build_kernel(void);
	void load_kernel(void);
	cl::Program _program;

	// Kernel I/O. Using cl:Buffer for now.
	// Need to create a derived class that will use SVM.
	// Jobs run in their own thread, so that the GPU doesn't block us.
	typedef struct
	{
		// This struct is access in a different thread the main thread.
		// Thus, we keep a pointer to the Value, so that it does not
		// get dtored before we access it.
		ValuePtr _kvec;
		cl::Kernel _kern;
		size_t _vecdim;
		ValueSeq _flovecs;
		OpenclFloatValuePtr _outvec;
	} job_t;
	void queue_job(const job_t&);
	async_caller<OpenclNode, job_t> _dispatch_queue;

	cl::Kernel get_kernel(ValuePtr) const;
	size_t get_vec_len(const ValueSeq&, bool&) const;
	ValuePtr get_floats(ValuePtr, cl::Kernel&, size_t&, size_t) const;
	ValueSeq make_vectors(ValuePtr, cl::Kernel&, size_t&) const;

	QueueValuePtr _qvp;
	virtual void open(const ValuePtr&);
	virtual void close(const ValuePtr&);
	virtual bool connected(void) const;
	virtual ValuePtr read(void) const;
	virtual ValuePtr stream(void) const;
	virtual void write_one(const ValuePtr&);
	virtual void do_write(const ValuePtr&);

public:
	OpenclNode(const std::string&&);
	OpenclNode(Type t, const std::string&&);
	virtual ~OpenclNode();


	static Handle factory(const Handle&);
};

NODE_PTR_DECL(OpenclNode)
#define createOpenclNode CREATE_DECL(OpenclNode)

/** @}*/
} // namespace opencog

extern "C" {
void opencog_opencl_init(void);
};

#endif // _OPENCOG_OPENCL_NODE_H
