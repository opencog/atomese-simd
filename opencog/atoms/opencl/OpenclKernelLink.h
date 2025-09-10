/*
 * opencog/atoms/opencl/OpenclKernelLink.h
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

#ifndef _OPENCOG_OPENCL_KERNEL_LINK_H
#define _OPENCOG_OPENCL_KERNEL_LINK_H

#include <opencog/atoms/opencl/opencl-headers.h>
#include <opencog/atoms/base/Link.h>

namespace opencog
{

/** \addtogroup grp_atomspace
 *  @{
 */

/**
 * OpenclKernelLinks hold an ordered vector of doubles.
 */
class OpenclKernelLink
	: public Link
{
	friend class OpenclNode;

protected:
	bool _have_kernel;
	cl::Kernel _kernel;

	cl::Kernel get_kernel(void);

	const std::string& get_kern_name (void) const;

public:
	// The argument is on stack very nearly 100% of the time,
	// so using the move ctor does not seem to make any sense.
	// OpenclKernelLink(HandleSeq&&, Type=OPENCL_KERNEL_LINK);
	OpenclKernelLink(HandleSeq, Type=OPENCL_KERNEL_LINK);
	virtual ~OpenclKernelLink();

	static Handle factory(const Handle&);
};

LINK_PTR_DECL(OpenclKernelLink)
#define createOpenclKernelLink CREATE_DECL(OpenclKernelLink)

/** @}*/
} // namespace opencog

#endif // _OPENCOG_OPENCL_KERNEL_LINK_H
