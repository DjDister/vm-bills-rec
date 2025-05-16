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

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET main PROPERTY CXX_STANDARD 20)
endif()


add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets
        $<TARGET_FILE_DIR:main>/assets
    COMMENT "Copying assets directory"
)


# kopiuje pliki z folderu assets do folderu build
file(COPY ${CMAKE_SOURCE_DIR}/assets DESTINATION ${CMAKE_BINARY_DIR}/Debug)
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/src)


set(CMAKE_FIND_DEBUG_MODE TRUE)
message(STATUS "Toolchain file: ${CMAKE_TOOLCHAIN_FILE}")

find_package(OpenCV REQUIRED)
target_link_libraries(main PRIVATE ${OpenCV_LIBS})