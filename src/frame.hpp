// Column-oriented table standing in for the pandas DataFrame the Python used.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace era {

// Numeric columns live in `num` keyed by name so the imputation and feature
// assembly loops stay name-driven, exactly as they were in features.py.
// Missing numeric values are NaN, which reproduces the NaN propagation the
// pandas `.fillna()` chains relied on.
struct Frame {
    std::vector<int64_t> player_id;
    std::vector<std::string> season_year;
    std::vector<std::string> player_name;
    std::vector<std::string> birth_date;
    std::vector<std::string> position;
    std::vector<std::string> era;  // filled by engineer_features
    std::map<std::string, std::vector<double>> num;

    // Longest identity column, so a frame carrying only some of them - as the
    // unit tests build - still reports its row count.
    std::size_t size() const {
        return std::max({player_id.size(), season_year.size(), position.size(), era.size()});
    }
};

}  // namespace era
