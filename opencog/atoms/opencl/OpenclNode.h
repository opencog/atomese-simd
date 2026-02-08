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
	friend class OpenclDataValue;
	friend class OpenclJobValue;

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
	const cl::Device& get_device(void) { return _device; }

	// Program loading and compilation.
	void build_program(void);
	void load_program(void);
	cl::Program _program;
	const cl::Program& get_program(void) { return _program; }

	// Binary caching for faster startup.
	std::string get_cache_dir(void) const;
	std::string get_cache_path(const std::string& src) const;
	std::string compute_hash(const std::string& data) const;
	bool load_cached_binary(const std::string& cache_path);
	void save_binary_to_cache(const std::string& cache_path);

	// List of interfaces provided by the program.
	// Its a bunch of kernels, described in Atomese.
	HandleMap _kernel_interfaces;

	// Execution context.
	// (TODO: I guess we could have several of these per device!?)
	cl::Context _context;
	const cl::Context& get_context(void) { return _context; }

	// Async I/O queue to the execution context.
	// (TODO: I guess we could have several of these per context!?)
	cl::CommandQueue _queue;
	cl::CommandQueue& get_queue(void) { return _queue; }

	cl::Event _event_handler;
	cl::Event& get_handler(void) { return _event_handler; }

	// Jobs run in their own thread, so that the GPU doesn't block us.
	void queue_job(const ValuePtr&);
	async_caller<OpenclNode, ValuePtr> _dispatch_queue;

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
