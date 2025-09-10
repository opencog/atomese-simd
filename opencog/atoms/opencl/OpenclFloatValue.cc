/*
 * opencog/atoms/opencl/OpenclFloatValue.cc
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
#include <opencog/atoms/value/ValueFactory.h>
#include <opencog/atoms/opencl/OpenclFloatValue.h>
#include <opencog/opencl/types/atom_types.h>

using namespace opencog;

OpenclFloatValue::OpenclFloatValue(size_t sz) :
	FloatValue(OPENCL_FLOAT_VALUE)
{
	_value.resize(sz);
}

OpenclFloatValue::OpenclFloatValue(const std::vector<double>& v) :
	FloatValue(OPENCL_FLOAT_VALUE, v)
{
}

OpenclFloatValue::OpenclFloatValue(std::vector<double>&& v) :
	FloatValue(OPENCL_FLOAT_VALUE, std::move(v))
{
}

// Right now, we will read from the GPU *every time*. Is this the
// right thing to do? Well, as envisioned in the Value subsystem
// design five years ago, the answer is "yes". Is this still the
// right answer? So far, seems to still be "yes". ...
void OpenclFloatValue::update(void) const
{
	if (not _have_ctxt) return;
	fetch_buffer();
}

void OpenclFloatValue::set_arg(cl::Kernel& kern, size_t pos)
{
	kern.setArg(pos, _buffer);
}

// ==============================================================

// Adds factory when the library is loaded.
DEFINE_VALUE_FACTORY(OPENCL_FLOAT_VALUE,
                     createOpenclFloatValue, size_t)
DEFINE_VALUE_FACTORY(OPENCL_FLOAT_VALUE,
                     createOpenclFloatValue, std::vector<double>)
