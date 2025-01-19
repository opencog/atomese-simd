/*
 * opencog/atoms/opencl/OpenclFloatVector.h
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

#ifndef _OPENCOG_OPENCL_FLOAT_VECTOR_H
#define _OPENCOG_OPENCL_FLOAT_VECTOR_H

#include <vector>
#include <opencog/atoms/value/StreamValue.h>

namespace opencog
{

/** \addtogroup grp_atomspace
 *  @{
 */

/**
 * OpenclFloatVectors hold an ordered vector of doubles.
 */
class OpenclFloatVector
	: public StreamValue
{
protected:
	virtual void update() const;

	OpenclFloatVector(Type t) : StreamValue(t) {}
	OpenclFloatVector(Type t, const std::vector<double>& v) : StreamValue(t, v) {}
public:
	OpenclFloatVector(void);
	OpenclFloatVector(const std::vector<double>&);

	virtual ~OpenclFloatVector() {}

	const std::vector<double>& value() const { update(); return _value; }
	size_t size() const { return _value.size(); }

	/** Returns true if two values are equal. */
	// virtual bool operator==(const Value&) const;
};

typedef std::shared_ptr<const OpenclFloatVector> OpenclFloatVectorPtr;
static inline OpenclFloatVectorPtr OpenclFloatVectorCast(const ValuePtr& a)
	{ return std::dynamic_pointer_cast<const OpenclFloatVector>(a); }

static inline const ValuePtr ValueCast(const OpenclFloatVectorPtr& fv)
{
	return std::shared_ptr<Value>(fv, (Value*) fv.get());
}

template<typename ... Type>
static inline std::shared_ptr<OpenclFloatVector> createOpenclFloatVector(Type&&... args) {
	return std::make_shared<OpenclFloatVector>(std::forward<Type>(args)...);
}

/** @}*/
} // namespace opencog

#endif // _OPENCOG_OPENCL_FLOAT_VECTOR_H
