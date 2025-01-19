
ADD_SUBDIRECTORY (scaffolding)
ADD_SUBDIRECTORY (types)

add_guile_extension(SCM_CONFIG opencl "opencog-ext-path-opencl")

add_guile_module(FILES
	opencl.scm
	${CMAKE_CURRENT_BINARY_DIR}/types/opencl_types.scm
	MODULE_DESTINATION "${GUILE_SITE_DIR}/opencog/opencl"
	DEPENDS opencl_atom_types
)

# ------------------------------------------------------------------
