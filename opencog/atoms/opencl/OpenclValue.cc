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
	_context(nullptr),
	_bytevec(nullptr)
{
}

OpenclValue::~OpenclValue()
{
	_context = nullptr;
}

void OpenclValue::set_context(const cl::Context& ctxt)
{
/*
	if (_context != nullptr)
		throw RuntimeException(TRACE_INFO,
			"Context already set!");
*/

	_context = ctxt;
}

void OpenclValue::from_gpu(size_t nbytes)
{
	// FIXME. probably OK if its already set!?
	// maybe should mark CL_MEM_READ_WRITE ???
/*
	if (nullptr != _bytevec)
		throw RuntimeException(TRACE_INFO,
			"Bytevec already set!");
*/

	_bytevec = cl::Buffer(_context, CL_MEM_READ_WRITE, nbytes);
}

void OpenclValue::to_gpu(size_t nbytes, void* vec)
{
	// FIXME. probably OK if its already set!?
/*
	if (nullptr != _bytevec)
		throw RuntimeException(TRACE_INFO,
			"Bytevec already set!");
*/

	_bytevec = cl::Buffer(_context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		nbytes, vec);
}

// ==============================================================
