/*
 * opencog/atoms/opencl/OpenclStream.h
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

#ifndef _OPENCOG_OPENCL_STREAM_H
#define _OPENCOG_OPENCL_STREAM_H

#include <opencog/atoms/sensory/OutputStream.h>

namespace opencog
{

/** \addtogroup grp_atomspace
 *  @{
 */

/**
 * OpenclStreams hold an ordered vector of doubles.
 */
class OpenclStream
	: public OutputStream
{
protected:
	OpenclStream(Type);
	void init(void);
	void halt(void) const;
	virtual void update() const;

	Handle _description;
	void do_describe(void);

public:
	OpenclStream(void);
	virtual ~OpenclStream();

	virtual ValuePtr describe(AtomSpace*, bool);
	virtual ValuePtr write_out(AtomSpace*, bool, const Handle&);
};

typedef std::shared_ptr<const OpenclStream> OpenclStreamPtr;
static inline OpenclStreamPtr OpenclStreamCast(const ValuePtr& a)
	{ return std::dynamic_pointer_cast<const OpenclStream>(a); }

static inline const ValuePtr ValueCast(const OpenclStreamPtr& fv)
{
	return std::shared_ptr<Value>(fv, (Value*) fv.get());
}

template<typename ... Type>
static inline std::shared_ptr<OpenclStream> createOpenclStream(Type&&... args) {
	return std::make_shared<OpenclStream>(std::forward<Type>(args)...);
}

/** @}*/
} // namespace opencog

extern "C" {
void opencog_opencl_init(void);
};

#endif // _OPENCOG_OPENCL_STREAM_H
