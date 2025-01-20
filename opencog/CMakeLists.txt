
IF (HAVE_OPENCL)
	# This declaration must happen before subdirectory traversal.
	DECLARE_GUILE_CONFIG_TARGET(SCM_CONFIG "opencog opencl-config" "OPENCL_TEST")
ENDIF (HAVE_OPENCL)

ADD_SUBDIRECTORY (atoms)

# ------------------------------------------------------------------
IF (HAVE_OPENCL)
	ADD_SUBDIRECTORY (opencl)

	# Testing framework boilerplate.
	WRITE_GUILE_CONFIG(${GUILE_BIN_DIR}/opencog/opencl-config.scm
		SCM_CONFIG TRUE)
	WRITE_GUILE_CONFIG(${GUILE_BIN_DIR}/opencog/opencl-config-installable.scm
		SCM_CONFIG FALSE)
	INSTALL(FILES ${GUILE_BIN_DIR}/opencog/opencl-config-installable.scm
		DESTINATION ${GUILE_SITE_DIR}/opencog RENAME opencl-config.scm)
ENDIF (HAVE_OPENCL)

# ------------------------------------------------------------------
IF (HAVE_CUDA)
	ADD_SUBDIRECTORY (cuda)
ENDIF (HAVE_CUDA)

# ------------------------------------------------------------------
