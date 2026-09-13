#pragma once

#include "image_types.hpp"

#include <string>

namespace XL {

// Persists completed inspection groups to SQLite and owns the mapping from
// in-memory result structures to the storage schema. It does not own threads,
// queues, camera hardware or detection policy.
class InspectionRepository {
public:
    InspectionRepository(
        std::string database_file,
        std::string output_root,
        std::string run_id,
        std::string device_id);

    bool save(const QuadFrameResult& group_result) const;

private:
    std::string mDatabaseFile;
    std::string mOutputRoot;
    std::string mRunId;
    std::string mDeviceId;
};

} // namespace XL
