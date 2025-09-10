/*
 * opencog/atoms/opencl/OpenclKernelNode.cc
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
#include <opencog/atoms/base/Link.h>
#include <opencog/atoms/base/Node.h>
#include <opencog/atoms/core/NumberNode.h>
#include <opencog/atoms/value/FloatValue.h>
#include <opencog/atoms/value/LinkValue.h>
#include <opencog/atoms/value/StringValue.h>
#include <opencog/atoms/value/VoidValue.h>
#include <opencog/atoms/value/ValueFactory.h>

#include <opencog/opencl/types/atom_types.h>
#include <opencog/sensory/types/atom_types.h>
#include "OpenclFloatValue.h"
#include "OpenclKernelNode.h"

using namespace opencog;

OpenclKernelNode::OpenclKernelNode(const std::string&& str) :
	Node(OPENCL_KERNEL_NODE, std::move(str))
{
}

OpenclKernelNode::OpenclKernelNode(Type t, const std::string&& str) :
	Node(t, std::move(str))
{
	if (not nameserver().isA(t, OPENCL_KERNEL_NODE))
		throw RuntimeException(TRACE_INFO,
			"Expecting OpenclKernelNode, got %s\n", to_string().c_str());
}

OpenclKernelNode::~OpenclKernelNode()
{
}

// ==============================================================

#if 0

/// Unwrap kernel name.
const std::string&
OpenclKernelNode::get_kern_name (ValuePtr vp) const
{
	if (vp->is_atom() and HandleCast(vp)->is_executable())
		vp = HandleCast(vp)->execute();

	if (vp->is_node())
		return HandleCast(vp)->get_name();

	if (vp->is_type(STRING_VALUE))
		return StringValueCast(vp)->value()[0];

	throw RuntimeException(TRACE_INFO,
		"Expecting Value with kernel name, got %s\n",
		vp->to_string().c_str());
}

cl::Kernel
OpenclKernelNode::get_kernel (ValuePtr kvec) const
{
	// Unpack kernel name.
	std::string kern_name;

	if (kvec->is_type(SECTION))
		kern_name = get_kern_name(HandleCast(kvec)->getOutgoingAtom(0));
	else
	if (kvec->is_type(SECTION_VALUE))
	{
		const ValueSeq& vsq = LinkValueCast(kvec)->value();
		kern_name = get_kern_name(vsq[0]);
	}
	else
		throw RuntimeException(TRACE_INFO,
			"Unknown data type: got %s\n", kvec->to_string().c_str());

	// XXX TODO this will throw exception if user mis-typed the
	// kernel name. We should catch this and print a friendlier
	// error message.
	return cl::Kernel(_program, kern_name.c_str());
}
#endif

// ==============================================================

// Adds factory when library is loaded.
DEFINE_NODE_FACTORY(OpenclKernelNode, OPENCL_KERNEL_NODE);

// ====================================================================
