
if(NOT "/home/zgojcic/code/DriveSim_AI/lib/external/.cache/numpyeigen/numpyeigen-download-prefix/src/numpyeigen-download-stamp/numpyeigen-download-gitinfo.txt" IS_NEWER_THAN "/home/zgojcic/code/DriveSim_AI/lib/external/.cache/numpyeigen/numpyeigen-download-prefix/src/numpyeigen-download-stamp/numpyeigen-download-gitclone-lastrun.txt")
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/home/zgojcic/code/DriveSim_AI/lib/external/.cache/numpyeigen/numpyeigen-download-prefix/src/numpyeigen-download-stamp/numpyeigen-download-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/home/zgojcic/code/DriveSim_AI/lib/external/numpyeigen"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/home/zgojcic/code/DriveSim_AI/lib/external/numpyeigen'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git" -c http.sslVerify=false clone --no-checkout --config "advice.detachedHead=false" "https://github.com/fwilliams/numpyeigen.git" "numpyeigen"
    WORKING_DIRECTORY "/home/zgojcic/code/DriveSim_AI/lib/external"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/fwilliams/numpyeigen.git'")
endif()

execute_process(
  COMMAND "/usr/bin/git" -c http.sslVerify=false checkout 7252ee0d8ae51414bbca3e0445ecfc285930b285 --
  WORKING_DIRECTORY "/home/zgojcic/code/DriveSim_AI/lib/external/numpyeigen"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: '7252ee0d8ae51414bbca3e0445ecfc285930b285'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git" -c http.sslVerify=false submodule update --recursive --init 
    WORKING_DIRECTORY "/home/zgojcic/code/DriveSim_AI/lib/external/numpyeigen"
    RESULT_VARIABLE error_code
    )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/home/zgojcic/code/DriveSim_AI/lib/external/numpyeigen'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/home/zgojcic/code/DriveSim_AI/lib/external/.cache/numpyeigen/numpyeigen-download-prefix/src/numpyeigen-download-stamp/numpyeigen-download-gitinfo.txt"
    "/home/zgojcic/code/DriveSim_AI/lib/external/.cache/numpyeigen/numpyeigen-download-prefix/src/numpyeigen-download-stamp/numpyeigen-download-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/home/zgojcic/code/DriveSim_AI/lib/external/.cache/numpyeigen/numpyeigen-download-prefix/src/numpyeigen-download-stamp/numpyeigen-download-gitclone-lastrun.txt'")
endif()

