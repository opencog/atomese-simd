/*
 * opencog/atoms/opencl/OpenclJobValue.h
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

#ifndef _OPENCOG_OPENCL_JOB_VALUE_H
#define _OPENCOG_OPENCL_JOB_VALUE_H

#include <vector>
#include <opencog/atoms/opencl/opencl-headers.h>
#include <opencog/atoms/value/LinkValue.h>
#include <opencog/atoms/opencl/OpenclFloatValue.h>

namespace opencog
{

/** \addtogroup grp_atomspace
 *  @{
 */

/**
 * OpenclJobValues hold OpenCL kernels bound to thier arguments.
 */
class OpenclJobValue :
	public LinkValue
{
	friend class OpenclNode;

protected:
	OpenclJobValue(Type t) : LinkValue(t), _is_built(false) {}

	Handle _definition;
	cl::Kernel _kernel;
	size_t _dim;

	// Buffers created during build() that need uploading to the GPU.
	// Upload is deferred to upload_inputs(), which runs on the
	// dispatch thread, avoiding races on the shared command queue.
	std::vector<OpenclFloatValuePtr> _pending_uploads;

	// Handle to OpenclNode, stored for deferred build in dispatch thread.
	// This allows build() to be called from queue_job() instead of do_write(),
	// ensuring all OpenCL kernel object creation happens in a single thread.
	// This eliminates per-thread OpenCL initialization overhead in multi-
	// threaded environments like CogServer.
	Handle _opencl_node;
	bool _is_built;

	void set_opencl_node(const Handle& h) { _opencl_node = h; }
	const Handle& get_opencl_node(void) const { return _opencl_node; }
	bool is_built(void) const { return _is_built; }

	void build(const Handle&);
	void upload_inputs(const Handle&);
	void run(const Handle&);
	void check_signature(const Handle&, const Handle&, const ValueSeq&);

	const std::string& get_kern_name (void) const;
	bool get_vec_len(const ValueSeq&);
	ValuePtr get_floats(const Handle&, ValuePtr);
	ValueSeq make_vectors(const Handle&);

public:
	OpenclJobValue(Handle);
	virtual ~OpenclJobValue();
};

VALUE_PTR_DECL(OpenclJobValue);
CREATE_VALUE_DECL(OpenclJobValue);

/** @}*/
} // namespace opencog


#endif // _OPENCOG_OPENCL_JOB_VALUE_H
