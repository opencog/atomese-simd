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

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300

#if defined __has_include
	#if __has_include(<CL/opencl.hpp>)
		#include <CL/opencl.hpp>
	#else
		#include <CL/cl.hpp>
	#endif
#else
	#include <CL/opencl.hpp>
#endif

#include <opencog/atoms/value/QueueValue.h>
#include <opencog/atoms/sensory/StreamNode.h>

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

	// Kernel compilation
	void build_kernel(void);
	void load_kernel(void);
	cl::Context _context;
	cl::Program _program;
	cl::CommandQueue _queue;

	// kernel I/O. Using cl:Buffer for now.
	// Need to create a derived class that will use SVM
	size_t _vec_dim;
	std::vector<cl::Buffer> _invec;
	cl::Buffer _outvec;  // XXX FIXME assume only one output
	cl::Kernel _kernel;

	const std::string& get_kern_name(AtomSpace*, bool, ValuePtr);
	const std::vector<double>& get_floats(AtomSpace*, bool, ValuePtr);

	QueueValuePtr _qvp;
	virtual void open(const ValuePtr&);
	virtual void close(const ValuePtr&);
	virtual bool connected(void) const;
	virtual ValuePtr read(void) const;
	virtual ValuePtr update(void) const;
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
