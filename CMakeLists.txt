cmake_minimum_required(VERSION 3.20.6...3.22.6 FATAL_ERROR)

project(SagittalMeasureAssist)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org")
set(EXTENSION_CONTRIBUTORS "Masaki Sakamoto (Kyoto University)")
set(EXTENSION_DESCRIPTION "Manual landmark-based measurement of sagittal alignment parameters (PI/PT/SS/LL) on lateral spine X-ray.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/SagittalMeasureAssist.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/SagittalMeasureAssist/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SagittalMeasureAssist)
add_subdirectory(SagittalMeasureTrain)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
