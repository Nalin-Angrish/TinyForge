# Examples — each is a standalone firmware main that links TinyForge
# Like FlexNN/src/main.cpp, but for TinyForge runtime

if(TINYFORGE_BUILD_EXAMPLES)
  add_executable(anomaly_detection anomaly_detection/main.cpp)
  target_link_libraries(anomaly_detection PRIVATE TinyForge::TinyForge)
  target_include_directories(anomaly_detection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../generated
  )

  add_executable(gesture gesture/main.cpp)
  target_link_libraries(gesture PRIVATE TinyForge::TinyForge)
  target_include_directories(gesture PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../generated
  )
endif()
