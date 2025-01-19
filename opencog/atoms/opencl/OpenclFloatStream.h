/*
 * opencog/atoms/opencl/OpenclFloatStream.h
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

#ifndef _OPENCOG_OPENCL_FLOAT_STREAM_H
#define _OPENCOG_OPENCL_FLOAT_STREAM_H

#include <vector>
#include <opencog/atoms/value/StreamValue.h>
#include <opencog/opencl/types/atom_types.h>

namespace opencog
{

/** \addtogroup grp_atomspace
 *  @{
 */

/**
 * OpenclFloatStreams hold an ordered vector of doubles.
 */
class OpenclFloatStream
	: public StreamValue
{
protected:
	virtual void update() const {}

	OpenclFloatStream(Type t) : StreamValue(t) {}
	OpenclFloatStream(Type t, const std::vector<double>& v) : StreamValue(t, v) {}
public:
	OpenclFloatStream()
		: StreamValue(OPENCL_FLOAT_STREAM) {}
	OpenclFloatStream(const std::vector<double>& v)
		: StreamValue(OPENCL_FLOAT_STREAM, v) {}

	virtual ~OpenclFloatStream() {}

	const std::vector<double>& value() const { update(); return _value; }
	size_t size() const { return _value.size(); }

	/** Returns true if two values are equal. */
	virtual bool operator==(const Value&) const;
};

typedef std::shared_ptr<const OpenclFloatStream> OpenclFloatStreamPtr;
static inline OpenclFloatStreamPtr OpenclFloatStreamCast(const ValuePtr& a)
	{ return std::dynamic_pointer_cast<const OpenclFloatStream>(a); }

static inline const ValuePtr ValueCast(const OpenclFloatStreamPtr& fv)
{
	return std::shared_ptr<Value>(fv, (Value*) fv.get());
}

template<typename ... Type>
static inline std::shared_ptr<OpenclFloatStream> createOpenclFloatStream(Type&&... args) {
	return std::make_shared<OpenclFloatStream>(std::forward<Type>(args)...);
}

/** @}*/
} // namespace opencog

#endif // _OPENCOG_OPENCL_FLOAT_STREAM_H
