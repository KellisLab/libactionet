#[=============================================================================[
ConfigureR.cmake
================================================================================
Enable build modes for debugging library and interface wrappers.

This module assumes the following modules have been included:

    * ConfigureBLAS.cmake
    * ConfigureApple.cmake

This module conditionally defines the following:

``R_HOME``
    Location of system R installation. Used by ``CONFIGURE_R()``.

``LIBACTIONET_BUILD_R``
    If defined and true, activates 'ConfigureR' module.

The following variables control the behaviour of this module:

``TEST_BUILD_R``
    If defined and true, enables R build test mode activates 'ConfigureR' module.

]=============================================================================]

## Test build for R interface
macro(TEST_BUILD_R libtarget)
    set(LIBACTIONET_BUILD_R 1)
    execute_process(
            COMMAND sh -c "R RHOME"
            OUTPUT_VARIABLE R_HOME
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    ## Test Rcpp wrappers
    target_sources(${libtarget}
            PRIVATE wrappers_r/wr_action.cpp
            PRIVATE wrappers_r/wr_annotation.cpp
            PRIVATE wrappers_r/wr_decomposition.cpp
            PRIVATE wrappers_r/wr_network.cpp
            PRIVATE wrappers_r/wr_tools.cpp
            PRIVATE wrappers_r/wr_visualization.cpp
            PRIVATE wrappers_r/wr_experimental.cpp
    )
endmacro()