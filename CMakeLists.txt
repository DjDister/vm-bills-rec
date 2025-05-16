# CMakeList.txt : CMake project for GameOfLife, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("main")

# Add source to this project's executable.
add_executable (main "main.cpp" "main.h")
target_sources(main PRIVATE

)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET main PROPERTY CXX_STANDARD 20)
endif()




# kopiuje pliki z folderu assets do folderu build

target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/src)


set(CMAKE_FIND_DEBUG_MODE TRUE)
message(STATUS "Toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
