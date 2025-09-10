/*
 * opencog/atoms/opencl/OpenclFloatValue.h
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

#ifndef _OPENCOG_OPENCL_FLOAT_VALUE_H
#define _OPENCOG_OPENCL_FLOAT_VALUE_H

#include <vector>
#include <opencog/atoms/value/FloatValue.h>
#include <opencog/atoms/opencl/OpenclValue.h>

namespace opencog
{

/** \addtogroup grp_atomspace
 *  @{
 */

/**
 * OpenclFloatValues hold an ordered vector of doubles.
 */
class OpenclFloatValue
	: public FloatValue, public OpenclValue
{
protected:
	virtual void update() const;

	OpenclFloatValue(Type t) : FloatValue(t) {}
	OpenclFloatValue(Type t, const std::vector<double>& v) : FloatValue(t, v) {}
	virtual size_t reserve_size(void) const {
		return sizeof(double) * _value.size(); }
	virtual void* data(void) const { return _value.data(); }

public:
	OpenclFloatValue(size_t);
	OpenclFloatValue(const std::vector<double>&);
	OpenclFloatValue(std::vector<double>&&);

	virtual ~OpenclFloatValue() {}

	const std::vector<double>& value() const { update(); return _value; }
	size_t size() const { return _value.size(); }

	void set_arg(cl::Kernel&, size_t pos);
	void resize(size_t dim) { _value.resize(dim); }
};

VALUE_PTR_DECL(OpenclFloatValue);
CREATE_VALUE_DECL(OpenclFloatValue);

/** @}*/
} // namespace opencog

#endif // _OPENCOG_OPENCL_FLOAT_VALUE_H
