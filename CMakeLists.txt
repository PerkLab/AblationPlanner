cmake_minimum_required(VERSION 3.5)

project(AblationPlanner)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/AblationPlanner")
set(EXTENSION_CATEGORY "Extensions")
set(EXTENSION_CONTRIBUTORS "Alice Santilli (PerkLab)")
set(EXTENSION_DESCRIPTION "Planning tool for RF ablation")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/AblationPlanner.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/AblationPlanner/Screenshots/1.png")
set(EXTENSION_DEPENDS "SlicerRT") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(LiverAblation)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
