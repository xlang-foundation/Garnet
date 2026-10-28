# Qwen3.5 服务状态查询与推理的锁等待

Mechrevo 的 Cantor 主进程 29128 在本地进程任务复验中仍存活，但模型调用和状态查询持续超时，GPU 利用率为零。原始快照保存在活动 Cantor 存储的 `.deployment/Cantor-29128-local-inference.dmp`。多个线程等待 `GarnetAPI::ServeStatusJson` 的实例互斥锁。

原来的 C++ 推理入口持有 `ServingInstance::mutex`，再回调原生 XLang3 的 `cuda_serving.Model.infer_json`。健康查询也等待同一把普通 C++ 锁。Python 解码器执行期间需要让出并重新获得 VM 执行权；此时再由嵌入调用等待普通锁，会让状态查询阻塞推理的恢复。Qwen3.5 模型已有 VM 能正确处理等待的 `threading.RLock`，不应再在外层持有 C++ 锁跨越整个模型回调。

修复对 Qwen3.5 保留模型句柄快照后释放实例锁，再调用推理或状态方法。列举模型先复制注册表快照，避免持有注册表锁逐一执行模型回调。复用已加载模型和停止模型遵循同样的原则。模型关闭现在持有自身的 RLock，保证正在执行的推理完成后才释放图与解码器缓冲区。TensorRT 其他模型的实例串行锁保持原有行为。

`test/xlang3/serving_concurrency.py` 通过真实 Garnet 公共入口加载可控制的假解码器，不需要权重，也不执行真实推理。它让一次推理等待，同时查询状态、列举模型、复用模型并请求停止；停止必须等待推理，并禁止提前释放缓冲区。旧 DLL 失败于 `health query waited for inference`；修复 DLL 在本机通过，Mechrevo 上也通过。真实 CUDA 设备不存在时该用例明确跳过，不当作通过。

验证：`garnet_native_api`、`garnet_serving_python_defaults`、`garnet_serving_concurrency` 三项 CTest 通过。Mechrevo 使用相同 DLL 哈希 `37E053B4B4883DEF2E7A48DA860C87879EF2EF023296A00893A687D7CAE562FE` 通过并发用例。真实 Qwen3.5-2B 模型的独立运行和与 RTX 5090 联合调度仍须另行验收；此处不声称已经通过，也不声称模型速度提升。

另一个用例 `serving_loading_concurrency.py` 复现了冷启动占住注册表锁的问题：加载第二个模型时，第一个已就绪模型的健康查询被阻塞。修复在构造期间释放注册表锁，用独立的初始化记录防止同一模型重复加载；构造失败会清除该记录，允许重试。正在加载时停止操作返回明确的 `model_loading`，不声称已停止，也不删除其他已就绪模型。两项并发用例与上述 API/默认配置用例共四项 CTest 通过；包含冷启动修复的新 DLL 哈希为 `DB8514C0E60C41F2FFA67D93FFCA5EB4B1F7CE385BF738AC76E29A4BC1F4613A`。Mechrevo 本轮真实模型正在加载第一版锁修复，不因新增冷启动修复重新启动。
