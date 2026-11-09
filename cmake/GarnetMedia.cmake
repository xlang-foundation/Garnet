# Core media code has no codec SDK dependency. Providers build independently.
function(garnet_configure_media target)
    find_package(Threads REQUIRED)
    target_sources(${target} PRIVATE "${PROJECT_SOURCE_DIR}/src/media/media_registry.cpp")
    target_link_libraries(${target} PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
endfunction()
