/*
 * opencog/atoms/opencl/OpenclValue.h
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

#ifndef _OPENCOG_OPENCL_VALUE_H
#define _OPENCOG_OPENCL_VALUE_H

namespace opencog
{

/** \addtogroup grp_atomspace
 *  @{
 */

/**
 * OpenclValues hold cl::Buffers that are needed to talk to the GPU/
 * Formally, these inherit from Value, but we don't want to actually
 * do this in C++, because the diamond inhertaince pattern will kill
 * kill us. We do not actually need anything that class Value provides.
 * We only need the OpenCL infrastructure.
 */
class OpenclValue
{
protected:
	OpenclValue();
public:
	virtual ~OpenclValue();
};

/** @}*/
} // namespace opencog

#endif // _OPENCOG_OPENCL_VALUE_H
