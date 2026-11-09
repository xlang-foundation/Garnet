#ifdef GARNET_METAL_FUSED_MISTRAL_MLP
        fusedMlpSites_.assign(ops_.size(),0);
        if(!mtpProgram_&&!verificationTokens_&&inputs_.size()==6&&inputShapes_[0].size()==3&&
           inputShapes_[0][2]==4096&&outputParts_.size()==3&&residentOps_.size()==ops_.size()) {
            std::vector<FusedMlpUseNode> uses;uses.reserve(ops_.size());
            for(const auto& op:ops_){FusedMlpUseNode n;n.reads={canonical(op.a),canonical(op.b)};for(auto id:op.arguments)n.reads.push_back(canonical(id));n.writes={canonical(op.out)};if(op.auxiliaryOutput!=SIZE_MAX)n.writes.push_back(canonical(op.auxiliaryOutput));uses.push_back(std::move(n));}
            std::vector<size_t> observable{canonical(output_)};for(auto id:inputs_)observable.push_back(canonical(id));for(auto id:outputParts_)observable.push_back(canonical(id));
            auto projection=[&](const Op& op,size_t inner,size_t columns){return op.kind==Kind::MatMul&&op.transpose&&slots_[op.b].affineInt4.Valid()&&slots_[op.a].columns==inner&&slots_[op.out].columns==columns&&slots_[op.a].rows>=128&&slots_[op.a].rows<=512&&op.indexedTokenProjection&&!op.indexedDecodeProjection;};
            for(size_t i=0;i+3<ops_.size();++i){const auto& gate=ops_[i];const auto& up=ops_[i+1];const auto& silu=ops_[i+2];const auto& down=ops_[i+3];
                if(!projection(gate,4096,14336)||!projection(up,4096,14336)||!projection(down,14336,4096)||gate.a!=up.a||silu.kind!=Kind::Silu||silu.a!=gate.out||silu.b!=up.out||down.a!=silu.out||!residentOps_[i]||!residentOps_[i+1]||!residentOps_[i+2]||!residentOps_[i+3])continue;
                auto protectedIds=observable;protectedIds.push_back(canonical(gate.a));protectedIds.push_back(canonical(down.out));
                if(ProveFusedMlpUses(uses,i,{canonical(gate.out),canonical(up.out),canonical(silu.out)},protectedIds))fusedMlpSites_[i]=1;
            }
        }
#endif
