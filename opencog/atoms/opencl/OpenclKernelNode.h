/*
 * opencog/atoms/opencl/OpenclKernelNode.h
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

#ifndef _OPENCOG_OPENCL_KERNEL_NODE_H
#define _OPENCOG_OPENCL_KERNEL_NODE_H

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

#include <opencog/atoms/base/Node.h>

namespace opencog
{

/** \addtogroup grp_atomspace
 *  @{
 */

/**
 * OpenclKernelNodes hold an ordered vector of doubles.
 */
class OpenclKernelNode
	: public Node
{
protected:

	cl::Kernel _kern;
public:
	OpenclKernelNode(const std::string&&);
	OpenclKernelNode(Type t, const std::string&&);
	virtual ~OpenclKernelNode();

	static Handle factory(const Handle&);
};

NODE_PTR_DECL(OpenclKernelNode)
#define createOpenclKernelNode CREATE_DECL(OpenclKernelNode)

/** @}*/
} // namespace opencog

#endif // _OPENCOG_OPENCL_KERNEL_NODE_H
