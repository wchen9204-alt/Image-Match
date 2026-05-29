#include "core/pipeline_manager.h"

#include "core/config.h"
#include "utils/logger.h"

namespace fs = std::filesystem;

namespace ir {

std::shared_ptr<Registration>
PipelineManager::load(const fs::path& yaml_path) {
    try {
        const PipelineConfig cfg = Config::loadPipeline(yaml_path);
        auto reg = std::make_shared<Registration>();
        if (!reg->configure(cfg)) return nullptr;
        regs_.push_back(reg);
        return reg;
    } catch (const std::exception& e) {
        IR_LOG_ERROR("PipelineManager::load failed for ", yaml_path.string(),
                     ": ", e.what());
        return nullptr;
    }
}

int PipelineManager::loadAll(const std::vector<fs::path>& yaml_paths) {
    int ok = 0;
    for (const auto& p : yaml_paths) {
        if (load(p)) ++ok;
    }
    IR_LOG_INFO("PipelineManager loaded ", ok, " / ", yaml_paths.size(), " pipelines.");
    return ok;
}

std::shared_ptr<Registration>
PipelineManager::findByName(const std::string& name) const {
    for (const auto& r : regs_) {
        if (r && r->name() == name) return r;
    }
    return nullptr;
}

} // namespace ir
