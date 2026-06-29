// Diagnostic: dump every ONNX graph's input/output names + shapes from an RF3M package, so a test/host
// can build a beam with the exact tensor shapes the model expects.
#include "radfield3d-nn/model_io.h"
#include <onnxruntime_cxx_api.h>
#include <cstdio>

int main(int argc, char** argv)
{
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.rf3m>\n", argv[0]); return 2; }
    rfnn::io::V1::NamedGraphs graphs;
    try { graphs = rfnn::io::V1::ModelStore::read_graphs(argv[1]); }
    catch (const std::exception& e) { std::fprintf(stderr, "read_graphs failed: %s\n", e.what()); return 2; }

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "dump");
    Ort::AllocatorWithDefaultOptions alloc;
    for (auto& kv : graphs) {
        std::printf("== graph '%s' (%zu bytes) ==\n", kv.first.c_str(), kv.second.size());
        Ort::SessionOptions so;
        Ort::Session sess(env, kv.second.data(), kv.second.size(), so);
        for (size_t i = 0; i < sess.GetInputCount(); ++i) {
            auto name = sess.GetInputNameAllocated(i, alloc);
            auto sh = sess.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            std::printf("  IN  %-14s [", name.get());
            for (auto d : sh) std::printf("%lld,", (long long)d);
            std::printf("]\n");
        }
        for (size_t i = 0; i < sess.GetOutputCount(); ++i) {
            auto name = sess.GetOutputNameAllocated(i, alloc);
            auto sh = sess.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            std::printf("  OUT %-14s [", name.get());
            for (auto d : sh) std::printf("%lld,", (long long)d);
            std::printf("]\n");
        }
    }
    return 0;
}
