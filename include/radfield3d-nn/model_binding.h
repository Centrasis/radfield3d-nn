#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <radfield3d-nn/model_interface.h>

namespace radfield3dnn {

// Zero-copy I/O. The CALLER owns every buffer and binds one per interface flag; the model reads
// straight from the bound input memory and writes straight into the bound output memory. It never
// allocates a result, so a renderer allocates its field buffers once and re-runs at video rate with
// no transfer between the model and the buffer it draws from.
//
// Bound INPUTS are re-read on every run(), so updating one in place is picked up by the next
// inference — unless it was bound with convert_buffer, which snapshots it (see the predictors'
// bind_global_parameter).

// Which execution provider a model should run on — and therefore which memory it can bind without a
// copy. Requesting one is how a caller guarantees the session matches the buffers it already owns.
// There is deliberately no Vulkan backend: ONNX Runtime has no Vulkan EP.
enum class Backend : std::uint8_t { Cpu = 0, Cuda = 1, TensorRT = 2 };

// Parse a backend name ("cpu" / "cuda" / "tensorrt"). Defined HERE, not in the Python layer, so both
// languages accept exactly the same spellings and produce the same error.
Backend backend_from_string(const std::string& name);
std::string to_string(Backend);

enum class DeviceKind : std::uint8_t { Cpu = 0, Cuda = 1, Vulkan = 2 };
enum class ElementType : std::uint8_t { F32 = 0, F16 = 1 };

std::string to_string(DeviceKind);
std::string to_string(ElementType);
std::string to_string(ModelInput);
std::string to_string(ModelOutput);

std::size_t element_size(ElementType);

// A non-owning view of caller memory. The caller guarantees it outlives the binding and (for device
// memory) that its own writes are synchronised before run().
struct MemoryRef {
    void*        data      = nullptr;
    std::size_t  bytes     = 0;      // CAPACITY of the allocation, not the size currently in use
    DeviceKind   device    = DeviceKind::Cpu;
    ElementType  dtype     = ElementType::F32;
    int          device_id = 0;      // Cuda / Vulkan
};

// What the session can consume with NO copy: the execution provider's device and the precision its
// kernels read/write at. Every binding is checked against this.
struct SessionMemory {
    DeviceKind  device    = DeviceKind::Cpu;
    ElementType dtype     = ElementType::F32;
    int         device_id = 0;
    std::string provider  = "CPUExecutionProvider";
};

// A bound buffer does not match what the session can consume, and the caller did not ask for a
// conversion.
class BindingError : public std::runtime_error {
public:
    explicit BindingError(const std::string& what) : std::runtime_error(what) {}
};

// run() was called before the model was fully set up. Lists EVERY missing piece at once.
class IncompleteSetupError : public std::runtime_error {
public:
    explicit IncompleteSetupError(const std::string& what) : std::runtime_error(what) {}
};

// The diagnostic a strict bind emits. Deliberately long: a silent host<->device copy per inference
// is the easiest way to lose everything this API exists to provide, so the caller is told exactly
// what mismatched, what it would cost, and the three distinct ways out — including how to opt IN.
std::string describe_binding_mismatch(const std::string& flag_name,
                                      const MemoryRef& buffer,
                                      const SessionMemory& session,
                                      std::size_t required_elements);

}  // namespace radfield3dnn
