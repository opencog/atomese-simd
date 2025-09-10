/*
 * opencog/atoms/opencl/OpenclValue.cc
 *
 * Copyright (C) 2015 Linas Vepstas
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

#include <opencog/util/exceptions.h>
#include <opencog/atoms/opencl/OpenclValue.h>

using namespace opencog;

OpenclValue::OpenclValue(void) :
	_have_ctxt(false),
	_device{},
	_context{},
	_queue{},
	_buffer{}
{
}

OpenclValue::~OpenclValue()
{
	_context = {};
	_device = {};
}

void OpenclValue::set_context(const cl::Device& ocldev,
                              const cl::Context& ctxt)
{
	if (_have_ctxt and _context != ctxt)
		throw RuntimeException(TRACE_INFO,
			"Context already set!");

	_have_ctxt = true;
	_device = ocldev;
	_context = ctxt;
	_queue = cl::CommandQueue(_context, _device);

	size_t nbytes = reserve_size();
	_buffer = cl::Buffer(_context, CL_MEM_READ_WRITE, nbytes);
}

// ==============================================================
