#ifdef GARNET_METAL_FUSED_MISTRAL_MLP
        if(a.columns==4096&&o.columns==14336) {
            if(!impl_->fusedMlpLibrary) {
                if(![impl_->device supportsFamily:MTLGPUFamilyApple10])throw std::runtime_error("Fused MLP requires Apple10");
                auto options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion4_0;options.mathMode=MTLMathModeFast;
                NSString* text=[@"#define GARNET_AFFINE_PROFILE 1\n#define GARNET_AFFINE_INDEX32 1\n#define GARNET_AFFINE_FP16_COEFFICIENTS 1\n" stringByAppendingString:[NSString stringWithUTF8String:kAffineInt4ProfilePrelude]];
                for(const auto* code:{kPackedAffineShader,kFusedMistralMlpShader,kPackedAffineLayoutGuardShader})text=[text stringByAppendingString:[NSString stringWithUTF8String:code]];
                auto library=[impl_->device newLibraryWithSource:text options:options error:&error];if(!library)throw std::runtime_error("Cannot compile fused MLP: "+std::string(error.localizedDescription.UTF8String));
                auto function=[library newFunctionWithName:@"packed_affine_layout_guard"];auto probe=[impl_->device newComputePipelineStateWithFunction:function error:&error];
                if(!probe||probe.threadExecutionWidth!=32)throw std::runtime_error("Cannot prepare fused MLP ABI probe");
                auto status=[impl_->device newBufferWithLength:128 options:MTLResourceStorageModeShared];if(!status)throw std::runtime_error("Cannot allocate fused ABI status");std::fill_n((uint32_t*)status.contents,32,UINT32_MAX);
                auto command=[impl_->queue commandBuffer];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:probe];[encoder setBuffer:status offset:0 atIndex:0];[encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];
                if(command.status!=MTLCommandBufferStatusCompleted||!PackedAffineLayoutMatches((uint32_t*)status.contents,32))throw std::runtime_error("Fused MLP cooperative ABI mismatch");
                auto memory=MemoryStats();if(memory.recommended<=14680064+256*1024*1024||memory.allocated>memory.recommended-14680064-256*1024*1024)throw std::runtime_error("Fused half arena exceeds native device budget");
                auto scratch=Allocate(512,7168);Zero(scratch);impl_->fusedMlpHalf=std::move(scratch);impl_->fusedMlpLibrary=library;
            }
            bool disabled=false;[constants setConstantValue:&disabled type:MTLDataTypeBool atIndex:20];[constants setConstantValue:&disabled type:MTLDataTypeBool atIndex:21];
            for(size_t i=0;i<2;++i){auto name=i?@"fused_m256k128":@"fused_m128k128";auto function=[impl_->fusedMlpLibrary newFunctionWithName:name constantValues:constants error:&error];if(!function)throw std::runtime_error("Cannot specialize fused MLP");auto pipeline=[impl_->device newComputePipelineStateWithFunction:function error:&error];if(!pipeline||pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<256||pipeline.staticThreadgroupMemoryLength!=17408)throw std::runtime_error("Fused MLP pipeline resource mismatch");plan.fused[i]=pipeline;}
        }
#endif
