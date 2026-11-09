template<typename T> METAL_FUNC void strict_raw_coords(thread const T& value,device uint* out,uint lane){
 for(ushort i=0;i<value.get_capacity();++i){auto xy=value.get_multidimensional_index(i);out[(lane*16+i)*3]=uint(xy[0]);out[(lane*16+i)*3+1]=uint(xy[1]);out[(lane*16+i)*3+2]=uint(value.is_valid_element(i));}
}
kernel void strict_mpp_raw_coords(device uint* out [[buffer(0)]],uint lane [[thread_index_in_simdgroup]]){
 constexpr auto qkd=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 constexpr auto pvd=matmul2d_descriptor(16,32,16,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<qkd,execution_simdgroup> qk;matmul2d<pvd,execution_simdgroup> pv;
 auto a=qk.get_left_input_cooperative_tensor<float,float,float>();auto b=qk.get_right_input_cooperative_tensor<float,float,float>();
 auto c=qk.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();
 auto x=pv.get_left_input_cooperative_tensor<float,float,float>();auto y=pv.get_right_input_cooperative_tensor<float,float,float>();
 auto z=pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(x)>,metal::remove_addrspace_t<decltype(y)>,float>();
 strict_raw_coords(a,out,lane);strict_raw_coords(b,out+1536,lane);strict_raw_coords(c,out+3072,lane);strict_raw_coords(x,out+4608,lane);strict_raw_coords(y,out+6144,lane);strict_raw_coords(z,out+7680,lane);
}
