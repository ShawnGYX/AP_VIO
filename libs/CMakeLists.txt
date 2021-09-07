
# The order of modules is important!
set(eqviolib_dirs
    core
)
if (BUILD_VISUALISATION)
    list(APPEND eqviolib_dirs visualisation)
endif()

set(MODULE_INCLUDE_DIRS "")
set(MODULE_LIBS "")

foreach(module ${eqviolib_dirs})
    add_subdirectory(${module})
    list(APPEND MODULE_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/libs/${module}/include")
    list(APPEND MODULE_LIBS "eqviolib_${module}")
endforeach()

set(MODULE_INCLUDE_DIRS ${MODULE_INCLUDE_DIRS} PARENT_SCOPE)
set(MODULE_LIBS ${MODULE_LIBS} PARENT_SCOPE)
