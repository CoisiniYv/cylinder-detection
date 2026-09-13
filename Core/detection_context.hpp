#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace XL {

// Immutable-by-convention output context for one process run.
//
// Detection workers receive this explicitly instead of reaching into the
// process-global RuntimeState to discover where results should be written.
struct DetectionContext {
    std::filesystem::path output_root;
    std::string run_id;

    bool valid() const noexcept { return !run_id.empty(); }

    std::filesystem::path runOutputDir() const {
        return output_root / run_id;
    }

    std::filesystem::path groupOutputDir(std::uint64_t group_id) const {
        return runOutputDir() / std::to_string(group_id);
    }

    std::filesystem::path singleOutputDir() const {
        return runOutputDir() / "single";
    }
};

} // namespace XL
