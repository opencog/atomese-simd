/*
 * opencog/atoms/opencl/OpenclStream.h
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

#ifndef _OPENCOG_OPENCL_STREAM_H
#define _OPENCOG_OPENCL_STREAM_H

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

#include <opencog/atoms/sensory/OutputStream.h>

namespace opencog
{

/** \addtogroup grp_atomspace
 *  @{
 */

/**
 * OpenclStreams hold an ordered vector of doubles.
 */
class OpenclStream
	: public OutputStream
{
protected:
	OpenclStream(Type);
	void init(const std::string& url);
	void halt(void);
	virtual void update() const;

	// API description
	Handle _description;
	void do_describe(void);

	// URL specifying platform and device.
	std::string _uri;
	std::string _splat; // platform substring
	std::string _sdev;  // device substring
	std::string _filepath; // path to cl, clcpp or spv file

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

	AtomSpacePtr _out_as;
	Type _out_type;
	const std::string& get_kern_name(AtomSpace*, bool, ValuePtr);
	const std::vector<double>& get_floats(AtomSpace*, bool, ValuePtr);

	void write_one(AtomSpace*, bool, const ValuePtr&);

public:
	OpenclStream(const Handle&);
	OpenclStream(const std::string&);
	virtual ~OpenclStream();

	virtual ValuePtr describe(AtomSpace*, bool);
	virtual ValuePtr write_out(AtomSpace*, bool, const Handle&);
};

typedef std::shared_ptr<const OpenclStream> OpenclStreamPtr;
static inline OpenclStreamPtr OpenclStreamCast(const ValuePtr& a)
	{ return std::dynamic_pointer_cast<const OpenclStream>(a); }

static inline const ValuePtr ValueCast(const OpenclStreamPtr& fv)
{
	return std::shared_ptr<Value>(fv, (Value*) fv.get());
}

template<typename ... Type>
static inline std::shared_ptr<OpenclStream> createOpenclStream(Type&&... args) {
	return std::make_shared<OpenclStream>(std::forward<Type>(args)...);
}

/** @}*/
} // namespace opencog

extern "C" {
void opencog_opencl_init(void);
};

#endif // _OPENCOG_OPENCL_STREAM_H
