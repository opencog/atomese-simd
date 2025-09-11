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

#include <opencog/atoms/opencl/opencl-headers.h>
#include <opencog/atoms/value/LinkValue.h>

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
protected:
	OpenclJobValue(Type t) : LinkValue(t) {}

	cl::Kernel _kernel;

public:
	OpenclJobValue(ValueSeq&&);
	virtual ~OpenclJobValue();
};

VALUE_PTR_DECL(OpenclJobValue);
CREATE_VALUE_DECL(OpenclJobValue);

/** @}*/
} // namespace opencog


#endif // _OPENCOG_OPENCL_JOB_VALUE_H
