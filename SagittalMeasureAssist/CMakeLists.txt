#-----------------------------------------------------------------------------
set(MODULE_NAME SagittalMeasureAssist)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  assist_controller.py
  logic_angles.py
  logic_export.py
  ui_measure.py
  ui_export.py
  )

set(MODULE_PYTHON_RESOURCES
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

endif()
