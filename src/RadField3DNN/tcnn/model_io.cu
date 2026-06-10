#include <radfield3d-nn/model_io.h>
#include <tiny-cuda-nn/common.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace rfnn {
namespace tcnn {

    namespace {
        template <typename T> void wpod(std::ostream& o, const T& v) {
            o.write(reinterpret_cast<const char*>(&v), sizeof(T));
        }
        template <typename T> T rpod(std::istream& i) {
            T v; i.read(reinterpret_cast<char*>(&v), sizeof(T)); return v;
        }
        void wstr(std::ostream& o, const std::string& s) {
            wpod<uint32_t>(o, (uint32_t)s.size());
            o.write(s.data(), (std::streamsize)s.size());
        }
        std::string rstr(std::istream& i) {
            uint32_t n = rpod<uint32_t>(i);
            std::string s(n, '\0');
            i.read(&s[0], n);
            return s;
        }
        // raw weight bytes for one sub-model (device → host vector)
        std::vector<char> dev_to_host(const ::tcnn::network_precision_t* dev, size_t n_params) {
            std::vector<char> buf(n_params * sizeof(::tcnn::network_precision_t));
            if (n_params) CUDA_CHECK_THROW(cudaMemcpy(buf.data(), dev, buf.size(), cudaMemcpyDeviceToHost));
            return buf;
        }
    }  // namespace

    void ModelFactory::save(const std::string& path,
                            const ::tcnn::network_precision_t* encoder_weights_device,
                            const std::string& encoder_type, const std::string& encoder_hparams_json,
                            size_t encoder_n_params,
                            const ::tcnn::network_precision_t* predictor_weights_device,
                            const std::string& predictor_type, const std::string& predictor_hparams_json,
                            size_t predictor_n_params) {
        std::ofstream o(path, std::ios::binary);
        if (!o) throw std::runtime_error("ModelFactory::save: cannot open " + path);

        o.write(kMagic, sizeof(kMagic));         // "RFNNM\0"
        wpod<uint8_t>(o, kVersion);

        const uint64_t enc_bytes = (uint64_t)encoder_n_params * sizeof(::tcnn::network_precision_t);
        const uint64_t pred_bytes = (uint64_t)predictor_n_params * sizeof(::tcnn::network_precision_t);

        // metadata
        wstr(o, encoder_type);   wstr(o, encoder_hparams_json);   wpod<uint64_t>(o, enc_bytes);
        wstr(o, predictor_type); wstr(o, predictor_hparams_json); wpod<uint64_t>(o, pred_bytes);

        // payload
        auto enc = dev_to_host(encoder_weights_device, encoder_n_params);
        auto pred = dev_to_host(predictor_weights_device, predictor_n_params);
        o.write(enc.data(), (std::streamsize)enc.size());
        o.write(pred.data(), (std::streamsize)pred.size());
    }

    std::unique_ptr<CombinedRadiationModel> ModelFactory::load(const std::string& path) {
        std::ifstream i(path, std::ios::binary);
        if (!i) throw std::runtime_error("ModelFactory::load: cannot open " + path);

        char magic[sizeof(kMagic)];
        i.read(magic, sizeof(kMagic));
        if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0)
            throw std::runtime_error("ModelFactory::load: bad magic in " + path);
        const uint8_t ver = rpod<uint8_t>(i);
        if (ver != kVersion)
            throw std::runtime_error("ModelFactory::load: unsupported version " + std::to_string(ver));

        const std::string enc_type = rstr(i); const std::string enc_h = rstr(i);
        const uint64_t enc_bytes = rpod<uint64_t>(i);
        const std::string pred_type = rstr(i); const std::string pred_h = rstr(i);
        const uint64_t pred_bytes = rpod<uint64_t>(i);

        std::vector<char> enc_w(enc_bytes), pred_w(pred_bytes);
        i.read(enc_w.data(), (std::streamsize)enc_bytes);
        i.read(pred_w.data(), (std::streamsize)pred_bytes);

        size_t enc_np = 0, pred_np = 0;
        auto encoder = build_module_from_hparams(enc_h, enc_np);
        auto predictor = build_module_from_hparams(pred_h, pred_np);
        if (enc_np * sizeof(::tcnn::network_precision_t) != enc_bytes ||
            pred_np * sizeof(::tcnn::network_precision_t) != pred_bytes)
            throw std::runtime_error("ModelFactory::load: weight size mismatch (architecture changed?)");

        // stage weights on the device, then hand to the combined model (it copies
        // into its own owned buffers and set_params()).
        ::tcnn::GPUMemory<::tcnn::network_precision_t> enc_dev(enc_np), pred_dev(pred_np);
        if (enc_np)  CUDA_CHECK_THROW(cudaMemcpy(enc_dev.data(),  enc_w.data(),  enc_bytes,  cudaMemcpyHostToDevice));
        if (pred_np) CUDA_CHECK_THROW(cudaMemcpy(pred_dev.data(), pred_w.data(), pred_bytes, cudaMemcpyHostToDevice));

        return std::make_unique<CombinedRadiationModel>(
            std::move(encoder), enc_np, enc_dev.data(),
            std::move(predictor), pred_np, pred_dev.data());
    }

}  // namespace tcnn
}  // namespace rfnn
