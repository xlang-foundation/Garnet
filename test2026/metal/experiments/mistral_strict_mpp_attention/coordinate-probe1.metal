
// Diagnostic-only coordinate coverage for all six strict cooperative operands.
template<typename T> METAL_FUNC void strict_coop_count(thread const T& value,device atomic_uint* out,device atomic_uint* invalid,uint width,uint height){
 for(ushort i=0;i<value.get_capacity();++i)if(value.is_valid_element(i)){auto xy=value.get_multidimensional_index(i);if(xy[0]<0||xy[1]<0||uint(xy[0])>=width||uint(xy[1])>=height)atomic_fetch_add_explicit(invalid,1,memory_order_relaxed);else atomic_fetch_add_explicit(out+uint(xy[1])*width+uint(xy[0]),1,memory_order_relaxed);}
}
kernel void strict_mpp_coordinates(device atomic_uint* out [[buffer(0)]]){
 constexpr auto qkd=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 constexpr auto pvd=matmul2d_descriptor(16,32,16,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<qkd,execution_simdgroup> qk;matmul2d<pvd,execution_simdgroup> pv;
 auto a=qk.get_left_input_cooperative_tensor<float,float,float>();auto b=qk.get_right_input_cooperative_tensor<float,float,float>();
 auto c=qk.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();
 auto x=pv.get_left_input_cooperative_tensor<float,float,float>();auto y=pv.get_right_input_cooperative_tensor<float,float,float>();
 auto z=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(x)>,metal::remove_addrspace_t<decltype(y)>,float>();
 strict_coop_count(a,out,out+2560,16,16);strict_coop_count(b,out+256,out+2560,16,32);strict_coop_count(c,out+768,out+2560,32,16);
 strict_coop_count(x,out+1280,out+2560,16,16);strict_coop_count(y,out+1536,out+2560,32,16);strict_coop_count(z,out+2048,out+2560,32,16);
}
