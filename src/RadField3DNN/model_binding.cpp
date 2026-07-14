#include <radfield3d-nn/model_binding.h>

#include <cctype>
#include <sstream>

namespace radfield3dnn {

std::string to_string(Backend b) {
    switch (b) {
        case Backend::Cpu:      return "cpu";
        case Backend::Cuda:     return "cuda";
        case Backend::TensorRT: return "tensorrt";
    }
    return "?";
}

Backend backend_from_string(const std::string& name) {
    std::string k;
    for (char c : name) k += static_cast<char>(::tolower(static_cast<unsigned char>(c)));
    if (k == "cpu")      return Backend::Cpu;
    if (k == "cuda")     return Backend::Cuda;
    if (k == "tensorrt") return Backend::TensorRT;
    if (k == "vulkan")
        throw BindingError(
            "backend 'vulkan': ONNX Runtime has no Vulkan execution provider, so no graph can read "
            "or write Vulkan memory. Load with backend 'cuda' and bind CUDA buffers; the "
            "CUDA->Vulkan export path shares the result with a Vulkan image without a host copy.");
    throw BindingError("unknown backend '" + name + "' — expected 'cpu', 'cuda' or 'tensorrt'.");
}

std::string to_string(DeviceKind d) {
    switch (d) {
        case DeviceKind::Cpu:    return "CPU";
        case DeviceKind::Cuda:   return "CUDA";
        case DeviceKind::Vulkan: return "Vulkan";
    }
    return "?";
}

std::string to_string(ElementType t) {
    switch (t) {
        case ElementType::F32: return "float32";
        case ElementType::F16: return "float16";
    }
    return "?";
}

std::size_t element_size(ElementType t) {
    return t == ElementType::F16 ? 2u : 4u;
}

std::string to_string(ModelInput f) {
    switch (f) {
        case ModelInput::None:                 return "None";
        case ModelInput::Position:             return "Position";
        case ModelInput::QueryDirection:       return "QueryDirection";
        case ModelInput::TubeSpectrum:         return "TubeSpectrum";
        case ModelInput::BeamDirection:        return "BeamDirection";
        case ModelInput::SourceDistance:       return "SourceDistance";
        case ModelInput::SourceOrigin3D:       return "SourceOrigin3D";
        case ModelInput::BeamCollimation:      return "BeamCollimation";
        case ModelInput::PatientTranslation3D: return "PatientTranslation3D";
        case ModelInput::PatientRotation3D:    return "PatientRotation3D";
        case ModelInput::GeometryMap:          return "GeometryMap";
        case ModelInput::AnodeAngle:           return "AnodeAngle";
    }
    return "?";
}

std::string to_string(ModelOutput f) {
    switch (f) {
        case ModelOutput::None:             return "None";
        case ModelOutput::Flux:             return "Flux";
        case ModelOutput::Spectrum:         return "Spectrum";
        case ModelOutput::AngularFlux:      return "AngularFlux";
        case ModelOutput::Error:            return "Error";
        case ModelOutput::AirKerma:         return "AirKerma";
        case ModelOutput::SeparateChannels: return "SeparateChannels";
    }
    return "?";
}

std::string describe_binding_mismatch(const std::string& flag_name,
                                      const MemoryRef& buffer,
                                      const SessionMemory& session,
                                      std::size_t required_elements) {
    const bool device_mismatch = buffer.device != session.device
                              || (buffer.device != DeviceKind::Cpu && buffer.device_id != session.device_id);
    const bool dtype_mismatch  = buffer.dtype != session.dtype;
    // Capacity is an ELEMENT count question, not a byte one: a float16 buffer holding the right
    // number of values is big enough, it just needs converting.
    const std::size_t have_elems = buffer.bytes / element_size(buffer.dtype);
    const bool too_small         = have_elems < required_elements;

    std::ostringstream o;
    o << "RadField3DNN: refusing to bind " << flag_name << " — the buffer does not match what this "
         "session can consume.\n\n";
    o << "  bound buffer : " << to_string(buffer.device);
    if (buffer.device != DeviceKind::Cpu) o << " (device " << buffer.device_id << ")";
    o << ", " << to_string(buffer.dtype) << ", room for " << have_elems << " values\n";
    o << "  session needs: " << to_string(session.device);
    if (session.device != DeviceKind::Cpu) o << " (device " << session.device_id << ")";
    o << ", " << to_string(session.dtype) << ", " << required_elements << " values"
      << "   [execution provider: " << session.provider << "]\n\n";

    if (too_small) {
        o << "  * CAPACITY: the buffer is too small. No conversion can fix this — the model would\n"
             "    write past the end of your memory. It must hold at least " << required_elements
          << " values.\n"
             "    Buffers are bound by CAPACITY: size for the largest grid you will ever request,\n"
             "    then change the grid per frame with set_voxel_grid().\n\n";
    }
    if (device_mismatch) {
        o << "  * DEVICE: the buffer lives in " << to_string(buffer.device) << " memory but the "
             "session executes on\n    " << to_string(session.device) << " (" << session.provider
          << "). Using it means a full host<->device copy of this\n    buffer on EVERY inference — "
             "usually larger than the inference itself, and it removes\n    the entire benefit of "
             "binding your own memory.\n\n";
    }
    if (dtype_mismatch) {
        o << "  * PRECISION: the buffer is " << to_string(buffer.dtype) << " but the session's "
             "kernels read/write\n    " << to_string(session.dtype) << ". Using it means converting "
             "the whole buffer element-by-element on\n    EVERY inference, and (for outputs) losing "
             "precision on the way back.\n\n";
    }

    o << "  Resolve it in one of three ways — deliberately, not by accident:\n"
         "    1. Bind matching memory: allocate the buffer as " << to_string(session.device)
      << " / " << to_string(session.dtype) << ".\n"
         "       This is the only option that stays zero-copy.\n"
         "    2. Move the session to your memory: load the model with the backend that matches the\n"
         "       buffers you already have (load_rf3m(path, backend=\"cuda\"), ExecutionOptions).\n"
         "    3. Accept the cost explicitly: pass convert_buffer = true to this bind_* call.\n"
         "       The model then stages and converts this buffer. It is off by default precisely so\n"
         "       that a per-inference copy can never appear without you asking for it.\n";
    return o.str();
}

}  // namespace radfield3dnn
