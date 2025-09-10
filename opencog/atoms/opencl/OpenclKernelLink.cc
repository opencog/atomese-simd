/*
 * opencog/atoms/opencl/OpenclKernelLink.cc
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
#include <opencog/atoms/value/ValueFactory.h>

#include <opencog/opencl/types/atom_types.h>
#include <opencog/sensory/types/atom_types.h>
#include "OpenclKernelLink.h"

using namespace opencog;

OpenclKernelLink::OpenclKernelLink(HandleSeq oset, Type t) :
	Link(std::move(oset), t),
	_have_kernel(false),
	_kernel{}
{
	if (not nameserver().isA(t, OPENCL_KERNEL_LINK))
		throw RuntimeException(TRACE_INFO,
			"Expecting OpenclKernelLink, got %s\n", to_string().c_str());
}

OpenclKernelLink::~OpenclKernelLink()
{
}

// ==============================================================

cl::Kernel
OpenclKernelLink::get_kernel(cl::Program& proggy)
{
	if (_have_kernel) return _kernel;

#if 0
	// XXX TODO this will throw exception if user mis-typed the
	// kernel name. We should catch this and print a friendlier
	// error message.
	_kernel = cl::Kernel(proggy, get_name().c_str());
#endif

	_have_kernel = true;
	return _kernel;
}

// ==============================================================

// Adds factory when library is loaded.
DEFINE_LINK_FACTORY(OpenclKernelLink, OPENCL_KERNEL_LINK);

// ====================================================================
