#include "radfield3d-nn/device_radiation_field.h"

#include <cstring>
#include <limits>

namespace radfield3dnn {

namespace {
// Copy every channel and layer's host data from one radiation field to another, regardless of
// the concrete buffer type (DeviceVoxelBuffer vs VoxelGridBuffer) or voxel type (scalar,
// histogram, …). Both fields must already share the same geometry / voxel count.
void copy_field_data(const RadFiled3D::IRadiationField& src, RadFiled3D::IRadiationField& dst) {
    for (const auto& [channel_name, src_buf] : src.get_channels()) {
        auto dst_buf = dst.add_channel(channel_name);
        for (const auto& layer_name : src_buf->get_layers()) {
            const RadFiled3D::VoxelLayer& layer = src_buf->get_layer(layer_name);
            if (src_buf->get_voxel_count() == 0) continue;
            // Replicate the layer's voxel type/byte layout from a template voxel, then byte-copy
            // the raw data buffer (get_bytes() = data bytes per voxel for any voxel type).
            const RadFiled3D::IVoxel* tmpl = layer.get_voxel_flat_raw(0);
            dst_buf->add_custom_layer_unsafe(layer_name, tmpl, layer.get_unit());
            const size_t bytes = tmpl->get_bytes() * src_buf->get_voxel_count();
            std::memcpy(dst_buf->get_layer<char>(layer_name), layer.get_raw_data(), bytes);
        }
    }
}
}  // namespace

DeviceCartesianRadiationField::DeviceCartesianRadiationField(const glm::vec3& field_dimensions,
                                                             const glm::vec3& voxel_dimensions,
                                                             DeviceBackend backend)
    : voxel_dimensions_(voxel_dimensions),
      // Match upstream CartesianRadiationField's count rule exactly (RadiationField.cpp):
      // add float epsilon before the divide so an exact box/voxel ratio doesn't truncate.
      voxel_counts_(glm::uvec3((field_dimensions + glm::vec3(std::numeric_limits<float>::epsilon()))
                               / voxel_dimensions)),
      field_dimensions_(field_dimensions),
      backend_(backend) {}

std::shared_ptr<DeviceCartesianRadiationField> DeviceCartesianRadiationField::from_dims(
    std::array<int, 3> dims, float field_box_m, DeviceBackend backend) {
    const glm::vec3 field(field_box_m, field_box_m, field_box_m);
    // Voxel size derives from the box / resolution (CLAUDE.md rule) — never hardcoded.
    const glm::vec3 voxel(field_box_m / static_cast<float>(dims[0]),
                          field_box_m / static_cast<float>(dims[1]),
                          field_box_m / static_cast<float>(dims[2]));
    return std::make_shared<DeviceCartesianRadiationField>(field, voxel, backend);
}

std::shared_ptr<RadFiled3D::VoxelBuffer> DeviceCartesianRadiationField::add_channel(
    const std::string& channel_name) {
    auto buffer = std::make_shared<DeviceVoxelBuffer>(this->voxel_count());
    this->channels[channel_name] = buffer;
    return buffer;
}

std::shared_ptr<RadFiled3D::CartesianRadiationField>
DeviceCartesianRadiationField::to_host_field() const {
    auto host = std::make_shared<RadFiled3D::CartesianRadiationField>(field_dimensions_,
                                                                      voxel_dimensions_);
    copy_field_data(*this, *host);
    return host;
}

std::shared_ptr<DeviceCartesianRadiationField>
DeviceCartesianRadiationField::from_host_field(const RadFiled3D::CartesianRadiationField& src,
                                               DeviceBackend backend) {
    auto dev = std::make_shared<DeviceCartesianRadiationField>(
        src.get_field_dimensions(), src.get_voxel_dimensions(), backend);
    copy_field_data(src, *dev);
    return dev;
}

std::shared_ptr<RadFiled3D::IRadiationField> DeviceCartesianRadiationField::copy() const {
    auto field = std::make_shared<DeviceCartesianRadiationField>(
        field_dimensions_, voxel_dimensions_, backend_);
    for (const auto& channel : this->channels) {
        // Deep-copy the host data; GPU mirrors are NOT copied (they are per-context and
        // re-uploaded on demand by the export helpers).
        auto* raw = static_cast<DeviceVoxelBuffer*>(channel.second->copy());
        field->channels[channel.first] = std::shared_ptr<DeviceVoxelBuffer>(raw);
    }
    return field;
}

}  // namespace radfield3dnn
