# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(numpyeigen-download NONE)

include(ExternalProject)
ExternalProject_Add(numpyeigen-download
                    GIT_REPOSITORY;https://github.com/fwilliams/numpyeigen.git;GIT_TAG;7252ee0d8ae51414bbca3e0445ecfc285930b285
                    SOURCE_DIR          "/home/zgojcic/code/DriveSim_AI/lib/external/numpyeigen"
                    BINARY_DIR          "/home/zgojcic/code/DriveSim_AI/lib/build/temp.linux-x86_64-3.8/numpyeigen-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    TLS_VERIFY          OFF
)
