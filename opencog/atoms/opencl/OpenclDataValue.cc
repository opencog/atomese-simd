/*
 * opencog/atoms/opencl/OpenclDataValue.cc
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
#include <opencog/atoms/opencl/OpenclDataValue.h>

using namespace opencog;

OpenclDataValue::OpenclDataValue(void) :
	_have_buff(false),
	_queue{},
	_buffer{}
{
}

OpenclDataValue::~OpenclDataValue()
{
}

/// Set up info about the GPU for this instance.
void OpenclDataValue::set_context(const cl::Device& ocldev,
                                  const cl::Context& ctxt)
{
	if (_have_buff) return;

	_have_buff = true;
	_queue = cl::CommandQueue(ctxt, ocldev);

	size_t nbytes = reserve_size();
	_buffer = cl::Buffer(ctxt, CL_MEM_READ_WRITE, nbytes);
}

/// Synchronously send data to the GPU
void OpenclDataValue::send_buffer(void) const
{
	if (not _have_buff)
		throw RuntimeException(TRACE_INFO,
			"No buffer!");

	cl::Event event_handler;
	size_t nbytes = reserve_size();
	const void* bytes = data();

	_queue.enqueueWriteBuffer(_buffer, CL_TRUE, 0,
		nbytes, bytes, nullptr, &event_handler);
	event_handler.wait();
}

/// Synchronously get data from the GPU
void OpenclDataValue::fetch_buffer(void) const
{
	// No-op if not yet tied to GPU.
	if (not _have_buff) return;

	cl::Event event_handler;
	size_t nbytes = reserve_size();
	void* bytes = data();

	_queue.enqueueReadBuffer(_buffer, CL_TRUE, 0,
		nbytes, bytes, nullptr, &event_handler);
	event_handler.wait();
}

void OpenclDataValue::set_arg(cl::Kernel& kern, size_t pos)
{
	kern.setArg(pos, _buffer);
}

// ==============================================================
