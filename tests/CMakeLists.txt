# Tests — host unit tests for TinyForge runtime
add_executable(test_runtime test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE TinyForge::TinyForge)

target_include_directories(test_runtime PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../generated
)

add_test(NAME tinyforge_runtime_smoke COMMAND test_runtime)

# Future: add more tests that compare scalar vs CMSIS stub vs FlexNN float
# add_executable(test_dense ...)
# add_test(NAME tinyforge_dense COMMAND test_dense)
