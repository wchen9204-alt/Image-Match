#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <yaml-cpp/yaml.h>

#include "core/config.h"
#include "core/context.h"
#include "dataset/dataset_loader.h"
#include "interfaces/i_pipeline.h"

namespace ir::registration_app_helpers {

void loadDatasetNamingOptions(const YAML::Node& ds, DatasetLoader::Options& options);

void printSummary(const RegistrationContext& ctx, MethodFamily family);

std::shared_ptr<IPipeline> createPipelineForConfig(const PipelineConfig& cfg);

std::string sampleStemFromPaths(const std::filesystem::path& image1,
                                const std::filesystem::path& image2);

std::filesystem::path normalizeOutputBaseRoot(std::filesystem::path root);

void applyCompareOverrides(PipelineConfig& pipeline_cfg,
                           const YAML::Node& compare_cfg,
                           const YAML::Node& combination,
                           const std::filesystem::path& compare_yaml_dir,
                           const std::filesystem::path& output_root,
                           const std::string& label);

void writeRunSummaryFiles(const RegistrationContext& ctx,
                          const PipelineConfig& cfg,
                          const std::string& sample_name);

} // namespace ir::registration_app_helpers
