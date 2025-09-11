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
#include "OpenclDataValue.h"
#include "OpenclNode.h"

using namespace opencog;

OpenclDataValue::OpenclDataValue(void) :
	_have_buff(false),
	_buffer{},
	_read_queue{},
	_read_event{}
{
}

OpenclDataValue::~OpenclDataValue()
{
}

/// Set up info about the GPU for this instance.
void OpenclDataValue::set_context(const Handle& oclno)
{
	if (_have_buff) return;
	_have_buff = true;

	OpenclNodePtr onp = OpenclNodeCast(oclno);

	// We use two different queues. For reads, we used our own queue
	// and event handler. This allows data to be read async from the
	// writers, avoiding getting accidentally blocked on the main
	// OpenclNode::_queue, which used for running kernels, and might
	// be busy for a long time.
	_read_queue = cl::CommandQueue(onp->get_context(), onp->get_device());
	_read_event = cl::Event();

	size_t nbytes = reserve_size();
	_buffer = cl::Buffer(onp->get_context(), CL_MEM_READ_WRITE, nbytes);
}

/// Synchronously send data to the GPU
void OpenclDataValue::send_buffer(const Handle& oclno) const
{
	if (not _have_buff)
		throw RuntimeException(TRACE_INFO,
			"No buffer!");

	OpenclNodePtr onp = OpenclNodeCast(oclno);
	cl::CommandQueue& queue = onp->get_queue();
	cl::Event& event_handler = onp->get_handler();

	size_t nbytes = reserve_size();
	const void* bytes = data();

	queue.enqueueWriteBuffer(_buffer, CL_TRUE, 0,
		nbytes, bytes, nullptr, &event_handler);
	event_handler.wait();
}

/// Synchronously get data from the GPU
void OpenclDataValue::fetch_buffer(void) const
{
	// No-op if not yet tied to GPU.
	if (not _have_buff) return;

	size_t nbytes = reserve_size();
	void* bytes = data();

	_read_queue.enqueueReadBuffer(_buffer, CL_TRUE, 0,
		nbytes, bytes, nullptr, &_read_event);
	_read_event.wait();
}

// ==============================================================
