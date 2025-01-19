/*
 * opencog/atoms/opencl/OpenclNode.h
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

#ifndef _OPENCOG_OPENCL_NODE_H
#define _OPENCOG_OPENCL_NODE_H

#include <vector>
#include <opencog/atoms/sensory/SensoryNode.h>
#include <opencog/opencl/types/atom_types.h>

namespace opencog
{

/** \addtogroup grp_atomspace
 *  @{
 */

/**
 * OpenclNodes hold an ordered vector of doubles.
 */
class OpenclNode
	: public SensoryNode
{
protected:
	virtual void update() const {}

	OpenclNode(Type t, const std::string&& uri)
		: OpenclNode(t, std::move(uri)) {}
public:
	OpenclNode(const std::string&&);

	virtual ~OpenclNode() {}
};

typedef std::shared_ptr<const OpenclNode> OpenclNodePtr;
static inline OpenclNodePtr OpenclNodeCast(const ValuePtr& a)
	{ return std::dynamic_pointer_cast<const OpenclNode>(a); }

static inline const ValuePtr ValueCast(const OpenclNodePtr& fv)
{
	return std::shared_ptr<Value>(fv, (Value*) fv.get());
}

template<typename ... Type>
static inline std::shared_ptr<OpenclNode> createOpenclNode(Type&&... args) {
	return std::make_shared<OpenclNode>(std::forward<Type>(args)...);
}

/** @}*/
} // namespace opencog

extern "C" {
void opencog_opencl_init(void);
};

#endif // _OPENCOG_OPENCL_NODE_H
