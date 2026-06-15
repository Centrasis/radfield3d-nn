#pragma once
// DeviceCartesianRadiationField — a RadField3DNN-side Cartesian radiation field whose
// per-voxel data can be mirrored into a GPU backend buffer (Vulkan today; DirectML /
// CUDA / ROCm later) for hand-off to a renderer such as Unreal Engine 5.
//
// Upstream RadFiled3D hardcodes `CartesianRadiationField : public
// RadiationField<VoxelGridBuffer>` (RadiationField.hpp), so the buffer type is NOT
// templated there. Rather than fork upstream, we subclass the *templated* base
// `RadiationField<BufferT>` with our own `DeviceVoxelBuffer`. A DeviceVoxelBuffer is an
// ordinary host VoxelBuffer (so all the existing RadFiled3D layer/voxel machinery and the
// FieldStore serializer keep working) plus an opaque, per-layer GPU handle that a backend
// export helper attaches. The header stays backend-agnostic: no Vulkan/CUDA types leak in
// (the handle is type-erased), so callers that only need the geometry/host data never pull
// in a GPU SDK.
//
// Geometry rule: a field box is 1.0 m, so voxel size = box / resolution. Construct
// from `dims` + the box; never hardcode a voxel size.

#include <array>
#include <map>
#include <memory>
#include <string>

#include <glm/vec3.hpp>
#include <RadFiled3D/RadiationField.hpp>
#include <RadFiled3D/VoxelBuffer.hpp>

namespace radfield3dnn {

// Channel/layer names that predict_into_field writes and the export helpers read. Kept here
// (backend-neutral) so neither side has to depend on a GPU backend header to agree on them.
inline constexpr const char* kPredictionChannel = "prediction";
inline constexpr const char* kFluxLayer         = "flux";
inline constexpr const char* kSpectrumLayer     = "spectrum";

// Which GPU backend a field's voxel data is (or can be) mirrored into. `Host` keeps data
// in CPU memory only (the default; identical to a plain RadFiled3D field). The non-Host
// backends are attached lazily by their export helper — the field never creates a GPU
// context of its own (the caller, e.g. UE5, owns it).
enum class DeviceBackend {
    Host,
    Vulkan,
    // DirectML,  // Windows (D3D12) — future
    // CUDA,      // NVIDIA          — future
    // ROCm,      // AMD             — future
};

// Opaque, type-erased GPU buffer handle. A backend (e.g. vulkan_field.cpp) stores its own
// concrete object (VkBuffer + VkDeviceMemory + size) behind this via a custom deleter, so
// this header carries no backend dependency. Lifetime is shared with the field.
using DeviceBufferHandle = std::shared_ptr<void>;

// A host VoxelBuffer that can additionally carry one opaque GPU buffer handle per layer.
// The host data is the source of truth; a handle is the mirror a backend uploads into.
class DeviceVoxelBuffer : public RadFiled3D::VoxelBuffer {
public:
    explicit DeviceVoxelBuffer(size_t voxel_count) : RadFiled3D::VoxelBuffer(voxel_count) {}

    // Attach / replace the GPU mirror for a named layer (called by an export helper).
    void set_device_handle(const std::string& layer, DeviceBufferHandle handle) {
        gpu_handles_[layer] = std::move(handle);
    }

    // The GPU mirror for a layer, or nullptr if it has not been uploaded yet.
    DeviceBufferHandle get_device_handle(const std::string& layer) const {
        auto it = gpu_handles_.find(layer);
        return it == gpu_handles_.end() ? nullptr : it->second;
    }

    bool has_device_handle(const std::string& layer) const {
        return gpu_handles_.find(layer) != gpu_handles_.end();
    }

private:
    std::map<std::string, DeviceBufferHandle> gpu_handles_;
};

// A Cartesian radiation field mirroring CartesianRadiationField's geometry API, backed by
// DeviceVoxelBuffer channels. Drop-in for read paths that only call get_voxel_counts() /
// get_field_dimensions() / get_voxel_dimensions(); additionally exposes the chosen backend.
class DeviceCartesianRadiationField : public RadFiled3D::RadiationField<DeviceVoxelBuffer> {
public:
    // Mirror of CartesianRadiationField(field_dimensions, voxel_dimensions).
    DeviceCartesianRadiationField(const glm::vec3& field_dimensions,
                                  const glm::vec3& voxel_dimensions,
                                  DeviceBackend backend = DeviceBackend::Host);

    // Convenience: build from an integer resolution and the (cubic) field box edge in
    // metres. voxel size = box / dims, per the repo's resolution rule.
    static std::shared_ptr<DeviceCartesianRadiationField> from_dims(
        std::array<int, 3> dims, float field_box_m = 1.0f,
        DeviceBackend backend = DeviceBackend::Host);

    const std::string& get_typename() const override {
        static const std::string name = "DeviceCartesianRadiationField";
        return name;
    }

    std::shared_ptr<RadFiled3D::VoxelBuffer> add_channel(const std::string& channel_name) override;

    std::shared_ptr<RadFiled3D::IRadiationField> copy() const override;

    // Download to a plain host-resident CartesianRadiationField: copies geometry + every
    // channel/layer's host data (any voxel type). GPU mirrors are not transferred — the device
    // field's host data is the source of truth, so sync a GPU-only result back via the backend
    // helper (e.g. download_floats) before calling this.
    std::shared_ptr<RadFiled3D::CartesianRadiationField> to_host_field() const;

    // Upload: build a DeviceCartesianRadiationField mirroring a host CartesianRadiationField
    // (copies geometry + channel/layer data). GPU mirrors are attached later by an export helper.
    static std::shared_ptr<DeviceCartesianRadiationField> from_host_field(
        const RadFiled3D::CartesianRadiationField& src,
        DeviceBackend backend = DeviceBackend::Host);

    const glm::vec3&  get_voxel_dimensions() const { return voxel_dimensions_; }
    const glm::uvec3& get_voxel_counts()     const { return voxel_counts_; }
    const glm::vec3&  get_field_dimensions() const { return field_dimensions_; }
    DeviceBackend     get_backend()          const { return backend_; }

    size_t voxel_count() const {
        return static_cast<size_t>(voxel_counts_.x) * voxel_counts_.y * voxel_counts_.z;
    }

private:
    const glm::vec3  voxel_dimensions_;
    const glm::uvec3 voxel_counts_;
    const glm::vec3  field_dimensions_;
    const DeviceBackend backend_;
};

}  // namespace radfield3dnn
