
if(GARNET_METAL_MISTRAL_DECODE_REUSE)
    add_executable(garnet_mistral_n4_decode_test test2026/metal/mistral_n4_decode_test.cpp)
    target_link_libraries(garnet_mistral_n4_decode_test PRIVATE garnet_metal)
    add_test(NAME metal_mistral_n4_decode_reference COMMAND garnet_mistral_n4_decode_test)
    set_tests_properties(metal_mistral_n4_decode_reference PROPERTIES TIMEOUT 240)
endif()
