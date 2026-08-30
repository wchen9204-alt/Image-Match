#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "core/config.h"
#include "core/context.h"
#include "core/result.h"
#include "dataset/dataset_loader.h"

namespace ir::registration_app_helpers {

void loadDatasetNamingOptions(const YAML::Node& ds, DatasetLoader::Options& options);

void printSummary(const RegistrationContext& ctx, MethodFamily family);

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

void writeBatchHtmlReport(const std::filesystem::path& pipeline_root,
                          const std::vector<std::string>& sample_names,
                          const std::vector<RegistrationResult>& results,
                          const std::filesystem::path& assets_root = {});

} // namespace ir::registration_app_helpers

